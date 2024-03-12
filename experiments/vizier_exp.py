from vizier import service
from vizier.service import clients
from vizier.service import pyvizier as vz
import pandas as pd
import os

from create_models import create_model
from utils.constants import (
    # Accelerators
    SVD,
    QUANTIZATION,
    PRUNING,

    # Models
    GPT2,
    GPT2_MEDIUM,

    # Parameters
    NUM_BLOCKS_GPT2,
    NUM_BLOCKS_GPT2_MEDIUM,

    # Datasets
    BILLSUM,
    OPENWEBTEXT,
    SQUAD,
    SST2,
)


def run_vizier_experiment(study, dataset_name: str, num_experiments: int = 10):
    # Run the experiments multiple times
    print("Running vizier...")
    for i in range(num_experiments):
        print("Experiment " + str(i))
        suggestions = study.suggest(count=1)
        print("suggestions: ", suggestions)
        for (j, suggestion) in enumerate(suggestions):
            print(j, suggestion)
            params = suggestion.parameters
            learning_rate = params["learning_rate"]

            model_name = f"GPT2_small_gated_{i}_{j}"

            print("learning rate: ", learning_rate)
            create_model(
                apply_gated_attention=True,
                model_name=model_name,
                model_type=GPT2,
                dataset_name=dataset_name,
                num_epochs=1,
                learning_rate=learning_rate,
            )

            log_csv = pd.read_csv(os.path.join("exp_logs"), model_name)
            eval_loss = log_csv["eval_loss"][0]
            
            suggestion.complete(vz.Measurement({'eval_loss': eval_loss}))
            print(f'Iteration {i}, suggestion {params} led to eval loss {eval_loss}')

    # Get the optimal results
    for optimal_trial in study.optimal_trials():
        optimal_trial = optimal_trial.materialize()
        print("Optimal Trial Suggestion and Objective:", optimal_trial.parameters,
              optimal_trial.final_measurement)


if __name__ == '__main__':
    # Algorithm, search space, and metrics.
    study_config = vz.StudyConfig(algorithm='GAUSSIAN_PROCESS_BANDIT')
    study_config.search_space.root.add_float_param('learning_rate', 1e-5, 1e-3)
    study_config.metric_information.append(vz.MetricInformation('eval_loss', goal=vz.ObjectiveMetricGoal.MINIMIZE))

    # Setup client and begin optimization. Vizier Service will be implicitly created.
    study = clients.Study.from_study_config(study_config, owner='lab', study_id='fast_transformer')
    print('Local SQL database file located at: ', service.VIZIER_DB_PATH)

    run_vizier_experiment(study, dataset_name=SST2, num_experiments=10)
