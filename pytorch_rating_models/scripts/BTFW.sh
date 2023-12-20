python3 pytorch_rating_models/train_rating.py \
    experiment=rate_prediction \
    trainer.max_epochs=10 \
    model.net.dropout_rate=0.5 \
    model.net.hidden_dim=512 \
    logger=csv \
    model.optimizer.lr=3e-5 \
    model.optimizer.weight_decay=0.0005 \
    task_name=wilson-target-tuned-notitle \
    data.target=wilson \
    data.use_scaler=False \
    model.net.use_sigmoid=True \
    model.net.output_size=1 \
    data.prepend_title=False \
    data.max_seq_len=512 \
    data.batch_size=8 \
    model.net.freeze=False \
    model.freeze_after=1