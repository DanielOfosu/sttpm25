from abc import ABC, abstractmethod
from typing import Tuple

import deepspeed
import models
import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
import utils
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from utils import RevIN, forecasting_metrics, lr_scheduler, plot


class Forecaster(pl.LightningModule, ABC):
    def __init__(
        self,
        config
    ):
        super().__init__()
        self.__dict__.update(config)
        self._inv_scaler = lambda x: x
        self.time_masked_idx = None
        self.linear_model = lambda x, *args, **kwargs: 0.0 
        if self.use_revin:
            self.revin = utils.RevIN.RevIN(num_features=self.d_yc)
        else:
            self.revin = lambda x, *args, **kwargs: x

        if self.use_seasonal_decomp:
            self.seasonal_decomp = utils.RevIN.SeriesDecomposition(kernel_size=25)
        else:
            self.seasonal_decomp = lambda x: (x, x.clone())

        self.save_hyperparameters()
        self.model = models.Model( #TODO: Check if the whole config can be passed here also to simplify code
            d_yc=self.d_yc,
            d_yt=self.d_yt,
            d_x=self.d_x,
            start_token_len=self.start_token_len,
            attn_factor=self.attn_factor,
            d_model=self.d_model,
            d_queries_keys=self.d_queries_keys,
            d_values=self.d_values,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_layers=self.d_layers,
            d_ff=self.d_ff,
            initial_downsample_convs=self.initial_downsample_convs,
            intermediate_downsample_convs=self.intermediate_downsample_convs,
            dropout_emb=self.dropout_emb,
            dropout_attn_out=self.dropout_attn_out,
            dropout_attn_matrix=self.dropout_attn_matrix,
            dropout_qkv=self.dropout_qkv,
            dropout_ff=self.dropout_ff,
            pos_emb_type=self.pos_emb_type,
            global_self_attn=self.global_self_attn,
            local_self_attn=self.local_self_attn,
            global_cross_attn=self.global_cross_attn,
            local_cross_attn=self.local_cross_attn,
            activation=self.activation,
            device=self.device,
            norm=self.norm,
            use_final_norm=self.use_final_norm,
            embed_method=self.embed_method,
            performer_attn_kernel=self.performer_kernel,
            performer_redraw_interval=self.performer_redraw_interval,
            attn_time_windows=self.attn_time_windows,
            use_shifted_time_windows=self.use_shifted_time_windows,
            time_emb_dim=self.time_emb_dim,
            verbose=True,
            null_value=self.null_value,
            pad_value=self.pad_value,
            max_seq_len=self.max_seq_len,
            recon_mask_skip_all=self.recon_mask_skip_all,
            recon_mask_max_seq_len=self.recon_mask_max_seq_len,
            recon_mask_drop_seq=self.recon_mask_drop_seq,
            recon_mask_drop_standard=self.recon_mask_drop_standard,
            recon_mask_drop_full=self.recon_mask_drop_full,
        )


    @property
    def train_step_forward_kwargs(self):
        return {"output_attn": False}

    @property
    def eval_step_forward_kwargs(self):
        return {"output_attn": False}

    def step(self, batch: Tuple[torch.Tensor], train: bool):
        kwargs = (
            self.train_step_forward_kwargs if train else self.eval_step_forward_kwargs
        )

        time_mask = self.time_masked_idx if train else None

        # compute all loss values
        loss_dict = self.compute_loss(
            batch=batch,
            time_mask=time_mask,
            forward_kwargs=kwargs,
        )

        forecast_out = loss_dict["forecast_out"]
        forecast_mask = loss_dict["forecast_mask"]
        *_, y_t = batch

        # compute prediction accuracy stats for logging
        stats = self._compute_stats(forecast_out, y_t, forecast_mask)

        stats["forecast_loss"] = loss_dict["forecast_loss"]
        stats["class_loss"] = loss_dict["class_loss"]
        stats["recon_loss"] = loss_dict["recon_loss"]

        # loss is a combination of forecasting, reconstruction and classification goals
        stats["loss"] = (
            loss_dict["forecast_loss"]
            + self.class_loss_imp * loss_dict["class_loss"]
            + self.recon_loss_imp * loss_dict["recon_loss"]
        )
        stats["acc"] = loss_dict["acc"]
        return stats

    def classification_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor]:

        labels = labels.view(-1).to(logits.device)
        d_y = labels.max() + 1

        logits = logits.view(-1, d_y)

        class_loss = F.cross_entropy(logits, labels)
        acc = torchmetrics.functional.accuracy(
            torch.softmax(logits, dim=1),
            labels,
        )
        return class_loss, acc

    def compute_loss(self, batch, time_mask=None, forward_kwargs=None):
        if forward_kwargs is None:
            forward_kwargs = {}
        x_c, y_c, x_t, y_t = batch

        forecast_out, recon_out, (logits, labels) = self(
            x_c, y_c, x_t, y_t, **forward_kwargs
        )

        # forecast (target seq prediction) loss
        forecast_loss, forecast_mask = self.forecasting_loss(
            outputs=forecast_out, y_t=y_t, time_mask=time_mask
        )

        if self.recon_loss_imp > 0:
            # reconstruction (masked? context seq prediction) loss
            recon_loss, recon_mask = self.forecasting_loss(
                outputs=recon_out, y_t=y_c, time_mask=None
            )
        else:
            recon_loss, recon_mask = -1.0, 0.0

        if self.class_loss_imp > 0:
            # space emb classification loss (detached)
            class_loss, acc = self.classification_loss(logits=logits, labels=labels)
        else:
            class_loss, acc = 0.0, -1.0

        return {
            "forecast_loss": forecast_loss,
            "class_loss": class_loss,
            "acc": acc,
            "forecast_out": forecast_out,
            "forecast_mask": forecast_mask,
            "recon_out": recon_out,
            "recon_loss": recon_loss,
            "recon_mask": recon_mask,
        }

    def nan_to_num(self, *inps):
        # override to let embedding handle NaNs
        return inps

    def forward_model_pass(self, x_c, y_c, x_t, y_t, output_attn=False):
        # set data to [batch, length, dim] format
        if len(y_c.shape) == 2:
            y_c = y_c.unsqueeze(-1)
        if len(y_t.shape) == 2:
            y_t = y_t.unsqueeze(-1)

        enc_x = x_c.to(self.device, dtype=torch.half)
        enc_y = y_c.to(self.device, dtype=torch.half)
        dec_x = x_t.to(self.device, dtype=torch.half)
        # zero out target sequence
        dec_y = torch.zeros_like(y_t).to(self.device).to(self.device, dtype=torch.half)
        
        if self.start_token_len > 0:
            # star token for the decoder from informer
            dec_y = torch.cat((y_c[:, -self.start_token_len :, :], dec_y), dim=1).to(
                self.device
            )
            dec_x = torch.cat((x_c[:, -self.start_token_len :, :], dec_x), dim=1)
        forecast_output, recon_output, (logits, labels), attn = self.model(
            enc_x=enc_x,
            enc_y=enc_y,
            dec_x=dec_x,
            dec_y=dec_y,
            output_attention=output_attn,
        )

        if output_attn:
            return forecast_output, recon_output, (logits, labels), attn
        return forecast_output, recon_output, (logits, labels)

    def validation_epoch_end(self, outs):
        total = 0
        count = 0
        for dict_ in outs:
            if "forecast_loss" in dict_:
                total += dict_["forecast_loss"].mean()
                count += 1
        avg_val_loss = total / count
        # manually tell scheduler it's the end of an epoch to activate
        # ReduceOnPlateau functionality from a step-based scheduler
        self.scheduler.step(avg_val_loss, is_end_epoch=True)

    def training_step_end(self, outs):
        self._log_stats("train", outs)
        self.scheduler.step()
        return {"loss": outs["loss"].mean()}
    
    def configure_optimizers(self):
        self.optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(
            self.parameters(),
            betas=(0.9, 0.995),
            lr=self.base_lr,
            weight_decay=self.l2_coeff, 
        )
         
        self.scheduler = lr_scheduler.WarmupReduceLROnPlateau(
            self.optimizer,
            init_lr=self.init_lr,
            peak_lr=self.base_lr,
            warmup_steps=self.warmup_steps,
            patience=3,
            factor=self.decay_factor,
        )
        return [self.optimizer], [self.scheduler]
    
    
    def set_null_value(self, val: float) -> None:
        self.null_value = val

    def set_inv_scaler(self, scaler) -> None:
        self._inv_scaler = scaler

    def set_scaler(self, scaler) -> None:
        self._scaler = scaler
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, train=True)

    def validation_step(self, batch, batch_idx):
        stats = self.step(batch, train=False)
        self.current_val_stats = stats
        return stats

    def test_step(self, batch, batch_idx):
        return self.step(batch, train=False)

    def _log_stats(self, section, outs):
        for key in outs.keys():
            stat = outs[key]
            if isinstance(stat, (np.ndarray, torch.Tensor)):
                stat = stat.mean()
            self.log(f"{section}/{key}", stat, sync_dist=True)

    def training_step_end(self, outs):
        self._log_stats("train", outs)
        return {"loss": outs["loss"].mean()}

    def validation_step_end(self, outs):
        self._log_stats("val", outs)
        return outs

    def test_step_end(self, outs):
        self._log_stats("test", outs)
        return {"loss": outs["loss"].mean()}

    def predict_step(self, batch, batch_idx):
        return self(*batch, **self.eval_step_forward_kwargs)

    def nan_to_num(self, *inps):
        return (torch.nan_to_num(i) for i in inps)

    def forward(
        self,
        x_c: torch.Tensor,
        y_c: torch.Tensor,
        x_t: torch.Tensor,
        y_t: torch.Tensor,
        **forward_kwargs,
    ) -> Tuple[torch.Tensor]:
        x_c, y_c, x_t, y_t = self.nan_to_num(x_c, y_c, x_t, y_t)
        _, pred_len, d_yt = y_t.shape

        y_c = self.revin(y_c, mode="norm")  # does nothing if use_revin = False

        seasonal_yc, trend_yc = self.seasonal_decomp(
            y_c
        )  # both are the original if use_seasonal_decomp = False
        preds, *extra = self.forward_model_pass(
            x_c, seasonal_yc, x_t, y_t, **forward_kwargs
        )
        baseline = self.linear_model(trend_yc, pred_len=pred_len, d_yt=d_yt)

        output = self.revin(preds + baseline, mode="denorm")
        return (output,) + tuple(extra) if extra else (output, )

    def _compute_stats(
        self, pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor
    ):
        pred = pred * mask
        true = torch.nan_to_num(true) * mask
        mask = mask.float()
        pred = pred.float()
        true = true.float()
        
        adj = mask.mean().cpu().numpy() + 1e-5
        pred = pred.detach().cpu().numpy()
        true = true.detach().cpu().numpy()
        scaled_pred = self._inv_scaler(pred)
        scaled_true = self._inv_scaler(true)
        stats = {
            "mape": forecasting_metrics.mape(scaled_true, scaled_pred) / adj,
            "mae": forecasting_metrics.mae(scaled_true, scaled_pred) / adj,
            "mse": forecasting_metrics.mse(scaled_true, scaled_pred) / adj,
            "smape": forecasting_metrics.smape(scaled_true, scaled_pred) / adj,
            "norm_mae": forecasting_metrics.mae(true, pred) / adj,
            "norm_mse": forecasting_metrics.mse(true, pred) / adj,
        }
        return stats

    def forecasting_loss(
        self, outputs: torch.Tensor, y_t: torch.Tensor, time_mask: int
    ) -> Tuple[torch.Tensor]:

        if self.null_value is not None:
            null_mask_mat = y_t != self.null_value
        else:
            null_mask_mat = torch.ones_like(y_t)

        # genuine NaN failsafe
        null_mask_mat *= ~torch.isnan(y_t)

        time_mask_mat = torch.ones_like(y_t)
        if time_mask is not None:
            time_mask_mat[:, time_mask:] = False

        full_mask = time_mask_mat * null_mask_mat
        forecasting_loss = self.loss_fn(y_t, outputs, full_mask)
        return forecasting_loss, full_mask

    def loss_fn(
        self, true: torch.Tensor, preds: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:

        true = torch.nan_to_num(true)

        if self.loss == "mse":
            loss = (mask * (true - preds)).square().sum() / max(mask.sum(), 1)
        elif self.loss == "mae":
            loss = torch.abs(mask * (true - preds)).sum() / max(mask.sum(), 1)
        elif self.loss == "smape":
            num = 2.0 * abs(preds - true)
            den = abs(preds.detach()) + abs(true) + 1e-5
            loss = 100.0 * (mask * (num / den)).sum() / max(mask.sum(), 1)
        else:
            raise ValueError(f"Unrecognized Loss Function : {self.loss}")
        return loss


