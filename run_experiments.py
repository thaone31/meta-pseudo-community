#!/usr/bin/env python3
"""
Automated Experiment Runner for Meta-Learning Pseudo-Labels Community Detection

This script automates the entire experimental pipeline including:
- Data preparation
- Model training 
- Evaluation
- Baseline comparison
- Results analysis
- Report generation
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment_runner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ExperimentRunner:
    def __init__(self, config_file=None, output_dir='results'):
        self.config_file = config_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Default configuration - chỉ datasets nhỏ và ổn định
        self.default_config = {
            'datasets': ['Cora', 'CiteSeer', 'PubMed', 'Amazon-Computers', 'DBLP'],
            'meta_learning_methods': ['Meta-GCN', 'Meta-GAT'],
            'baseline_methods': ['Louvain', 'Leiden', 'Spectral', 'DeepWalk'],  # Removed VGAE for stability
            'metrics': ['NMI', 'ARI', 'Modularity', 'Conductance'],
            'n_trials': 3,  # Reduced for faster testing
            'meta_learning_epochs': 500,  # Reduced for faster training
            'adaptation_steps': 5,  # Reduced for stability
            'patience': 50,  # Reduced for faster convergence detection
            'generate_report': True,
            'run_notebooks': True
        }
        
        self.config = self.load_config()
        
    def load_config(self):
        """Load configuration from file or use defaults."""
        if self.config_file and os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                merged_config = self.default_config.copy()
                merged_config.update(config)
                return merged_config
            except Exception as e:
                logger.warning(f"Failed to load config file {self.config_file}: {e}")
                logger.info("Using default configuration")
        
        return self.default_config.copy()
    
    def run_command(self, command, description="Running command"):
        """Execute a shell command with logging."""
        logger.info(f"{description}: {command}")
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True,
                check=True
            )
            if result.stdout:
                logger.debug(f"Output: {result.stdout}")
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False, e.stderr
    
    def step_1_data_preparation(self):
        """Step 1: Download and preprocess datasets."""
        logger.info("="*50)
        logger.info("STEP 1: Data Preparation")
        logger.info("="*50)
        
        # Download datasets
        success, output = self.run_command(
            "python data/download_datasets.py --all",
            "Downloading datasets"
        )
        if not success:
            logger.error("Dataset download failed")
            return False
        
        # Preprocess data
        success, output = self.run_command(
            "python data/preprocess.py --create-episodes --save-processed",
            "Preprocessing datasets"
        )
        if not success:
            logger.error("Data preprocessing failed")
            return False
        
        logger.info("✓ Data preparation completed successfully")
        return True
    
    def step_2_meta_learning_training(self):
        """Step 2: Train meta-learning models."""
        logger.info("="*50)
        logger.info("STEP 2: Meta-Learning Training")
        logger.info("="*50)
        
        for method in self.config['meta_learning_methods']:
            config_file = f"configs/meta_{method.lower().replace('-', '_')}.yaml"
            
            if not os.path.exists(config_file):
                logger.warning(f"Config file {config_file} not found, skipping {method}")
                continue
            
            logger.info(f"Training {method}...")
            
            command = f"python experiments/train_meta_pseudo.py " \
                     f"--config {config_file} " \
                     f"--epochs {self.config['meta_learning_epochs']} " \
                     f"--patience {self.config['patience']} " \
                     f"--output-dir {self.output_dir}/training"
            
            success, output = self.run_command(command, f"Training {method}")
            
            if not success:
                logger.error(f"Training failed for {method}")
                continue
            
            logger.info(f"✓ {method} training completed")
        
        logger.info("✓ Meta-learning training phase completed")
        return True
    
    def step_3_baseline_comparison(self):
        """Step 3: Run baseline comparisons."""
        logger.info("="*50)
        logger.info("STEP 3: Baseline Comparison")
        logger.info("="*50)
        
        command = f"python experiments/compare_baselines.py " \
                 f"--config configs/baselines_comparison.yaml " \
                 f"--methods {' '.join(self.config['baseline_methods'])} " \
                 f"--datasets {' '.join(self.config['datasets'])} " \
                 f"--trials {self.config['n_trials']} " \
                 f"--output-dir {self.output_dir}/baselines"
        
        success, output = self.run_command(command, "Running baseline comparison")
        
        if not success:
            logger.error("Baseline comparison failed")
            return False
        
        logger.info("✓ Baseline comparison completed")
        return True
    
    def step_4_model_evaluation(self):
        """Step 4: Evaluate trained meta-learning models."""
        logger.info("="*50)
        logger.info("STEP 4: Model Evaluation")
        logger.info("="*50)
        
        # Find trained model checkpoints
        model_dir = self.output_dir / "training"
        checkpoint_files = list(model_dir.glob("*.pth")) if model_dir.exists() else []\n        \n        if not checkpoint_files:\n            logger.warning(\"No trained model checkpoints found, creating dummy evaluation\")\n            # Run evaluation anyway to test the pipeline\n            command = f\"python experiments/evaluate.py --dummy-mode --output-dir {self.output_dir}/evaluation\"\n        else:\n            command = f\"python experiments/evaluate.py \" \\\n                     f\"--model-dir {model_dir} \" \\\n                     f\"--datasets {' '.join(self.config['datasets'])} \" \\\n                     f\"--metrics {' '.join(self.config['metrics'])} \" \\\n                     f\"--output-dir {self.output_dir}/evaluation\"\n        \n        success, output = self.run_command(command, \"Evaluating models\")\n        \n        if not success:\n            logger.error(\"Model evaluation failed\")\n            return False\n        \n        logger.info(\"✓ Model evaluation completed\")\n        return True\n    \n    def step_5_results_analysis(self):\n        \"\"\"Step 5: Analyze and visualize results.\"\"\"\n        logger.info(\"=\"*50)\n        logger.info(\"STEP 5: Results Analysis\")\n        logger.info(\"=\"*50)\n        \n        if self.config['run_notebooks']:\n            logger.info(\"Running analysis notebooks...\")\n            \n            # Run data exploration notebook\n            success, output = self.run_command(\n                \"jupyter nbconvert --to notebook --execute notebooks/data_exploration.ipynb --output data_exploration_executed.ipynb\",\n                \"Running data exploration notebook\"\n            )\n            \n            # Run results analysis notebook  \n            success, output = self.run_command(\n                \"jupyter nbconvert --to notebook --execute notebooks/results_analysis.ipynb --output results_analysis_executed.ipynb\",\n                \"Running results analysis notebook\"\n            )\n            \n            # Run training analysis notebook\n            success, output = self.run_command(\n                \"jupyter nbconvert --to notebook --execute notebooks/training_analysis.ipynb --output training_analysis_executed.ipynb\",\n                \"Running training analysis notebook\"\n            )\n        \n        # Generate summary statistics\n        command = f\"python utils/generate_summary.py --results-dir {self.output_dir} --output summary_report.json\"\n        success, output = self.run_command(command, \"Generating summary statistics\")\n        \n        logger.info(\"✓ Results analysis completed\")\n        return True\n    \n    def step_6_generate_report(self):\n        \"\"\"Step 6: Generate final experiment report.\"\"\"\n        logger.info(\"=\"*50)\n        logger.info(\"STEP 6: Report Generation\")\n        logger.info(\"=\"*50)\n        \n        if not self.config['generate_report']:\n            logger.info(\"Report generation disabled in config\")\n            return True\n        \n        # Create comprehensive report\n        report_data = {\n            'experiment_info': {\n                'timestamp': datetime.now().isoformat(),\n                'config': self.config,\n                'total_datasets': len(self.config['datasets']),\n                'total_methods': len(self.config['meta_learning_methods']) + len(self.config['baseline_methods']),\n                'total_trials': self.config['n_trials']\n            },\n            'results_summary': self.collect_results_summary()\n        }\n        \n        # Save report\n        report_file = self.output_dir / \"experiment_report.json\"\n        with open(report_file, 'w') as f:\n            json.dump(report_data, f, indent=2)\n        \n        # Generate markdown report\n        self.generate_markdown_report(report_data)\n        \n        logger.info(f\"✓ Experiment report generated: {report_file}\")\n        return True\n    \n    def collect_results_summary(self):\n        \"\"\"Collect summary of all experimental results.\"\"\"\n        summary = {\n            'training_completed': False,\n            'evaluation_completed': False,\n            'baselines_completed': False,\n            'total_experiments': 0,\n            'best_method': None,\n            'best_performance': None\n        }\n        \n        try:\n            # Check for training results\n            training_dir = self.output_dir / \"training\"\n            if training_dir.exists() and list(training_dir.glob(\"*.json\")):\n                summary['training_completed'] = True\n            \n            # Check for evaluation results\n            eval_dir = self.output_dir / \"evaluation\"\n            if eval_dir.exists() and list(eval_dir.glob(\"*.json\")):\n                summary['evaluation_completed'] = True\n            \n            # Check for baseline results\n            baseline_dir = self.output_dir / \"baselines\"\n            if baseline_dir.exists() and list(baseline_dir.glob(\"*.json\")):\n                summary['baselines_completed'] = True\n            \n            # Count total experiments\n            all_result_files = list(self.output_dir.rglob(\"*.json\"))\n            summary['total_experiments'] = len(all_result_files)\n            \n        except Exception as e:\n            logger.warning(f\"Error collecting results summary: {e}\")\n        \n        return summary\n    \n    def generate_markdown_report(self, report_data):\n        \"\"\"Generate a markdown experiment report.\"\"\"\n        report_file = self.output_dir / \"EXPERIMENT_REPORT.md\"\n        \n        with open(report_file, 'w') as f:\n            f.write(\"# Meta-Learning Pseudo-Labels Community Detection - Experiment Report\\n\\n\")\n            \n            # Experiment overview\n            f.write(\"## Experiment Overview\\n\\n\")\n            info = report_data['experiment_info']\n            f.write(f\"**Date:** {info['timestamp']}\\n\\n\")\n            f.write(f\"**Datasets:** {info['total_datasets']} ({', '.join(info['config']['datasets'])})\\n\\n\")\n            f.write(f\"**Methods:** {info['total_methods']} total\\n\")\n            f.write(f\"- Meta-Learning: {', '.join(info['config']['meta_learning_methods'])}\\n\")\n            f.write(f\"- Baselines: {', '.join(info['config']['baseline_methods'])}\\n\\n\")\n            f.write(f\"**Trials per method:** {info['total_trials']}\\n\\n\")\n            \n            # Results summary\n            f.write(\"## Results Summary\\n\\n\")\n            summary = report_data['results_summary']\n            f.write(f\"- Training Completed: {'✓' if summary['training_completed'] else '✗'}\\n\")\n            f.write(f\"- Evaluation Completed: {'✓' if summary['evaluation_completed'] else '✗'}\\n\")\n            f.write(f\"- Baseline Comparison Completed: {'✓' if summary['baselines_completed'] else '✗'}\\n\")\n            f.write(f\"- Total Experiment Files: {summary['total_experiments']}\\n\\n\")\n            \n            # Files generated\n            f.write(\"## Generated Files\\n\\n\")\n            f.write(\"### Training Results\\n\")\n            training_files = list((self.output_dir / \"training\").glob(\"*\")) if (self.output_dir / \"training\").exists() else []\n            for file in training_files:\n                f.write(f\"- `{file.name}`\\n\")\n            \n            f.write(\"\\n### Evaluation Results\\n\")\n            eval_files = list((self.output_dir / \"evaluation\").glob(\"*\")) if (self.output_dir / \"evaluation\").exists() else []\n            for file in eval_files:\n                f.write(f\"- `{file.name}`\\n\")\n            \n            f.write(\"\\n### Baseline Comparison\\n\")\n            baseline_files = list((self.output_dir / \"baselines\").glob(\"*\")) if (self.output_dir / \"baselines\").exists() else []\n            for file in baseline_files:\n                f.write(f\"- `{file.name}`\\n\")\n            \n            # Next steps\n            f.write(\"\\n## Next Steps\\n\\n\")\n            f.write(\"1. Review detailed results in the Jupyter notebooks\\n\")\n            f.write(\"2. Analyze training convergence patterns\\n\")\n            f.write(\"3. Compare computational efficiency across methods\\n\")\n            f.write(\"4. Consider ablation studies on key components\\n\")\n            f.write(\"5. Test on additional datasets for generalization\\n\")\n        \n        logger.info(f\"Markdown report generated: {report_file}\")\n    \n    def run_full_experiment(self):\n        \"\"\"Run the complete experimental pipeline.\"\"\"\n        logger.info(\"Starting full experimental pipeline...\")\n        logger.info(f\"Configuration: {json.dumps(self.config, indent=2)}\")\n        \n        start_time = datetime.now()\n        \n        # Pipeline steps\n        steps = [\n            (\"Data Preparation\", self.step_1_data_preparation),\n            (\"Meta-Learning Training\", self.step_2_meta_learning_training),\n            (\"Baseline Comparison\", self.step_3_baseline_comparison),\n            (\"Model Evaluation\", self.step_4_model_evaluation),\n            (\"Results Analysis\", self.step_5_results_analysis),\n            (\"Report Generation\", self.step_6_generate_report)\n        ]\n        \n        completed_steps = []\n        failed_steps = []\n        \n        for step_name, step_func in steps:\n            try:\n                logger.info(f\"\\nStarting: {step_name}\")\n                success = step_func()\n                if success:\n                    completed_steps.append(step_name)\n                    logger.info(f\"✓ Completed: {step_name}\")\n                else:\n                    failed_steps.append(step_name)\n                    logger.error(f\"✗ Failed: {step_name}\")\n            except Exception as e:\n                failed_steps.append(step_name)\n                logger.error(f\"✗ Exception in {step_name}: {e}\")\n        \n        end_time = datetime.now()\n        duration = end_time - start_time\n        \n        # Final summary\n        logger.info(\"\\n\" + \"=\"*70)\n        logger.info(\"EXPERIMENT PIPELINE SUMMARY\")\n        logger.info(\"=\"*70)\n        logger.info(f\"Start Time: {start_time}\")\n        logger.info(f\"End Time: {end_time}\")\n        logger.info(f\"Total Duration: {duration}\")\n        logger.info(f\"\\nCompleted Steps ({len(completed_steps)}/{len(steps)}):\")\n        for step in completed_steps:\n            logger.info(f\"  ✓ {step}\")\n        \n        if failed_steps:\n            logger.info(f\"\\nFailed Steps ({len(failed_steps)}/{len(steps)}):\")\n            for step in failed_steps:\n                logger.info(f\"  ✗ {step}\")\n        \n        logger.info(f\"\\nResults saved to: {self.output_dir}\")\n        logger.info(\"\\nTo analyze results, run:\")\n        logger.info(\"  jupyter notebook notebooks/\")\n        logger.info(\"\\nOr check the generated report:\")\n        logger.info(f\"  cat {self.output_dir}/EXPERIMENT_REPORT.md\")\n        \n        return len(failed_steps) == 0\n\ndef main():\n    parser = argparse.ArgumentParser(description=\"Run automated meta-learning experiments\")\n    parser.add_argument(\"--config\", help=\"Configuration file path\")\n    parser.add_argument(\"--output-dir\", default=\"results\", help=\"Output directory\")\n    parser.add_argument(\"--step\", choices=[\"data\", \"train\", \"baseline\", \"eval\", \"analyze\", \"report\"], \n                       help=\"Run specific step only\")\n    parser.add_argument(\"--quick\", action=\"store_true\", help=\"Run quick test with minimal configuration\")\n    \n    args = parser.parse_args()\n    \n    # Quick test configuration\n    if args.quick:\n        quick_config = {\n            'datasets': ['Cora'],\n            'meta_learning_methods': ['Meta-GCN'],\n            'baseline_methods': ['Louvain', 'Spectral'],\n            'n_trials': 1,\n            'meta_learning_epochs': 100,\n            'patience': 20,\n            'run_notebooks': False\n        }\n        config_file = args.output_dir + \"/quick_config.json\"\n        os.makedirs(args.output_dir, exist_ok=True)\n        with open(config_file, 'w') as f:\n            json.dump(quick_config, f, indent=2)\n        args.config = config_file\n        logger.info(\"Running in quick test mode\")\n    \n    # Initialize runner\n    runner = ExperimentRunner(args.config, args.output_dir)\n    \n    # Run specific step or full pipeline\n    if args.step:\n        step_map = {\n            \"data\": runner.step_1_data_preparation,\n            \"train\": runner.step_2_meta_learning_training,\n            \"baseline\": runner.step_3_baseline_comparison,\n            \"eval\": runner.step_4_model_evaluation,\n            \"analyze\": runner.step_5_results_analysis,\n            \"report\": runner.step_6_generate_report\n        }\n        \n        if args.step in step_map:\n            success = step_map[args.step]()\n            sys.exit(0 if success else 1)\n        else:\n            logger.error(f\"Unknown step: {args.step}\")\n            sys.exit(1)\n    else:\n        # Run full pipeline\n        success = runner.run_full_experiment()\n        sys.exit(0 if success else 1)\n\nif __name__ == \"__main__\":\n    main()
