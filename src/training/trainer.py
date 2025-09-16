import lightning as L
import mlflow
import mlflow.pytorch


def setup_mlflow():
    mlflow.set_tracking_uri("file:./experiments/mlruns")
    mlflow.set_experiment("mnist_experiments")
    mlflow.pytorch.autolog()


def train_model(model, data_module, config, description="", run_name=""):
    setup_mlflow()
    
    if run_name:
        mlflow.start_run(run_name=run_name)
    else:
        mlflow.start_run()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    print(f"Config: {config}")
    print(f"Description: {description}")
    
    # Log config parameters and model info
    mlflow.log_params(config)
    mlflow.log_param("total_parameters", total_params)
    if description:
        mlflow.set_tag("description", description)
    
    trainer = L.Trainer(
        max_epochs=config['max_epochs'],
        accelerator='auto',
        devices=1,
        deterministic=True
    )
    
    trainer.fit(model, data_module)
    
    val_results = trainer.validate(model, data_module)
    print(f"Final validation results: {val_results}")
    
    return trainer, val_results