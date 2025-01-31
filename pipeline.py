from kfp import dsl
from kfp import compiler
from kfp.registry import RegistryClient


# Define a component that processes data
@dsl.component
def process_data(data_size: int) -> str:
    import random

    processed_count = random.randint(1, data_size)
    return f"Processed {processed_count} records"


# Define a component that trains a model
@dsl.component
def train_model(learning_rate: float, epochs: int, input_data: str) -> float:
    import random

    accuracy = random.random()
    print(f"Training with {input_data}")
    print(f"Parameters: learning_rate={learning_rate}, epochs={epochs}")
    return accuracy


# Define a component that evaluates the model
@dsl.component
def evaluate_model(accuracy: float) -> str:
    threshold = 0.7
    result = "passed" if accuracy > threshold else "failed"
    return f"Model evaluation {result} with accuracy {accuracy:.2f}"


# Define the tutorial pipeline
@dsl.pipeline(
    name="Tutorial Pipeline",
    description="A simple pipeline for learning KFP basics",
)
def tutorial_pipeline(
    data_size: int = 1000, learning_rate: float = 0.01, epochs: int = 10
):
    # Process data
    data_op = process_data(data_size=data_size)

    # Train model using processed data
    train_op = train_model(
        learning_rate=learning_rate, epochs=epochs, input_data=data_op.output
    )

    # Evaluate the model
    evaluate_model(accuracy=train_op.output)


# Compile the pipeline
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=tutorial_pipeline, package_path="tutorial_pipeline.yaml"
    )

    client = RegistryClient(
        host="https://us-central1-kfp.pkg.dev/learn-vertex-pipelines/kubeflow"
    )
    templateName, versionName = client.upload_pipeline(
        file_name="tutorial_pipeline.yaml",
        tags=["v1", "latest"],
        extra_headers={"description": "Beans pipeline template."},
    )
