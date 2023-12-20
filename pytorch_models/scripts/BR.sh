python3 pytorch_models/train_rating.py \
    experiment=rate_prediction \
    trainer.max_epochs=10 \
    model.net.dropout_rate=0.5 \
    model.net.hidden_dim=512 \
    logger=csv \
    model.optimizer.lr=3e-5 \
    model.optimizer.weight_decay=0.0005 \
    task_name=raw-target-tuned-title \
    data.target=raw data.use_scaler=True \
    model.net.use_sigmoid=False \
    model.net.output_size=2 \
    data.prepend_title=True \
    data.max_seq_len=512 \
    data.batch_size=8 \
    model.net.freeze=False \
    model.freeze_after=1