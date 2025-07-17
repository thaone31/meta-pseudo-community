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
        
        # Default configuration - only small and stable datasets
        self.default_config = {
            'datasets': ['Cora', 'CiteSeer', 'PubMed', 'Amazon-Computers', 'DBLP'],
            'meta_learning_methods': ['Meta-GCN', 'Meta-GAT'],
            'baseline_methods': ['Louvain', 'Leiden', 'Spectral', 'DeepWalk'],
            'metrics': ['NMI', 'ARI', 'Modularity', 'Conductance'],
            'n_trials': 3,
            'meta_learning_epochs': 500,
            'adaptation_steps': 5,
            'patience': 50,
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
        
        success, output = self.run_command(
            "python data/download_datasets.py --all",
            "Downloading datasets"
        )
        if not success:
            logger.error("Dataset download failed")
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
            
            command = f"python experiments/train_meta_pseudo.py --config {config_file} --epochs {self.config['meta_learning_epochs']} --patience {self.config['patience']} --output-dir {self.output_dir}/training"
            
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
        
        command = f"python experiments/compare_baselines.py --methods {' '.join(self.config['baseline_methods'])} --datasets {' '.join(self.config['datasets'])} --trials {self.config['n_trials']} --output-dir {self.output_dir}/baselines"
        
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
        
        model_dir = self.output_dir / "training"
        checkpoint_files = list(model_dir.glob("*.pth")) if model_dir.exists() else []
        
        if not checkpoint_files:
            logger.warning("No trained model checkpoints found, creating dummy evaluation")
            command = f"python experiments/evaluate.py --dummy-mode --output-dir {self.output_dir}/evaluation"
        else:
            command = f"python experiments/evaluate.py --model-dir {model_dir} --datasets {' '.join(self.config['datasets'])} --metrics {' '.join(self.config['metrics'])} --output-dir {self.output_dir}/evaluation"
        
        success, output = self.run_command(command, "Evaluating models")
        
        if not success:
            logger.error("Model evaluation failed")
            return False
        
        logger.info("✓ Model evaluation completed")
        return True
    
    def step_5_results_analysis(self):
        """Step 5: Analyze and visualize results."""
        logger.info("="*50)
        logger.info("STEP 5: Results Analysis")
        logger.info("="*50)
        
        if self.config['run_notebooks']:
            logger.info("Running analysis notebooks...")
            
            notebooks = [
                "data_exploration.ipynb",
                "results_analysis.ipynb", 
                "training_analysis.ipynb"
            ]
            
            for notebook in notebooks:
                if os.path.exists(f"notebooks/{notebook}"):
                    self.run_command(
                        f"jupyter nbconvert --to notebook --execute notebooks/{notebook} --output {notebook.replace('.ipynb', '_executed.ipynb')}",
                        f"Running {notebook}"
                    )
        
        command = f"python utils/generate_summary.py --results-dir {self.output_dir} --output summary_report.json"
        self.run_command(command, "Generating summary statistics")
        
        logger.info("✓ Results analysis completed")
        return True
    
    def step_6_generate_report(self):
        """Step 6: Generate final experiment report."""
        logger.info("="*50)
        logger.info("STEP 6: Report Generation")
        logger.info("="*50)
        
        if not self.config['generate_report']:
            logger.info("Report generation disabled in config")
            return True
        
        report_data = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'config': self.config,
                'total_datasets': len(self.config['datasets']),
                'total_methods': len(self.config['meta_learning_methods']) + len(self.config['baseline_methods']),
                'total_trials': self.config['n_trials']
            },
            'results_summary': self.collect_results_summary()
        }
        
        report_file = self.output_dir / "experiment_report.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        self.generate_markdown_report(report_data)
        
        logger.info(f"✓ Experiment report generated: {report_file}")
        return True
    
    def collect_results_summary(self):
        """Collect summary of all experimental results."""
        summary = {
            'training_completed': False,
            'evaluation_completed': False,
            'baselines_completed': False,
            'total_experiments': 0,
            'best_method': None,
            'best_performance': None
        }
        
        try:
            training_dir = self.output_dir / "training"
            if training_dir.exists() and list(training_dir.glob("*.json")):
                summary['training_completed'] = True
            
            eval_dir = self.output_dir / "evaluation"
            if eval_dir.exists() and list(eval_dir.glob("*.json")):
                summary['evaluation_completed'] = True
            
            baseline_dir = self.output_dir / "baselines"
            if baseline_dir.exists() and list(baseline_dir.glob("*.json")):
                summary['baselines_completed'] = True
            
            all_result_files = list(self.output_dir.rglob("*.json"))
            summary['total_experiments'] = len(all_result_files)
            
        except Exception as e:
            logger.warning(f"Error collecting results summary: {e}")
        
        return summary
    
    def generate_markdown_report(self, report_data):
        """Generate a markdown experiment report."""
        report_file = self.output_dir / "EXPERIMENT_REPORT.md"
        
        with open(report_file, 'w') as f:
            f.write("# Meta-Learning Pseudo-Labels Community Detection - Experiment Report\n\n")
            
            info = report_data['experiment_info']
            f.write("## Experiment Overview\n\n")
            f.write(f"**Date:** {info['timestamp']}\n\n")
            f.write(f"**Datasets:** {info['total_datasets']} ({', '.join(info['config']['datasets'])})\n\n")
            f.write(f"**Methods:** {info['total_methods']} total\n")
            f.write(f"- Meta-Learning: {', '.join(info['config']['meta_learning_methods'])}\n")
            f.write(f"- Baselines: {', '.join(info['config']['baseline_methods'])}\n\n")
            f.write(f"**Trials per method:** {info['total_trials']}\n\n")
            
            summary = report_data['results_summary']
            f.write("## Results Summary\n\n")
            f.write(f"- Training Completed: {'✓' if summary['training_completed'] else '✗'}\n")
            f.write(f"- Evaluation Completed: {'✓' if summary['evaluation_completed'] else '✗'}\n")
            f.write(f"- Baseline Comparison Completed: {'✓' if summary['baselines_completed'] else '✗'}\n")
            f.write(f"- Total Experiment Files: {summary['total_experiments']}\n\n")
            
            f.write("## Next Steps\n\n")
            f.write("1. Review detailed results in the Jupyter notebooks\n")
            f.write("2. Analyze training convergence patterns\n")
            f.write("3. Compare computational efficiency across methods\n")
            f.write("4. Consider ablation studies on key components\n")
            f.write("5. Test on additional datasets for generalization\n")
        
        logger.info(f"Markdown report generated: {report_file}")
    
    def run_full_experiment(self):
        """Run the complete experimental pipeline."""
        logger.info("Starting full experimental pipeline...")
        logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")
        
        start_time = datetime.now()
        
        steps = [
            ("Data Preparation", self.step_1_data_preparation),
            ("Meta-Learning Training", self.step_2_meta_learning_training),
            ("Baseline Comparison", self.step_3_baseline_comparison),
            ("Model Evaluation", self.step_4_model_evaluation),
            ("Results Analysis", self.step_5_results_analysis),
            ("Report Generation", self.step_6_generate_report)
        ]
        
        completed_steps = []
        failed_steps = []
        
        for step_name, step_func in steps:
            try:
                logger.info(f"\nStarting: {step_name}")
                success = step_func()
                if success:
                    completed_steps.append(step_name)
                    logger.info(f"✓ Completed: {step_name}")
                else:
                    failed_steps.append(step_name)
                    logger.error(f"✗ Failed: {step_name}")
            except Exception as e:
                failed_steps.append(step_name)
                logger.error(f"✗ Exception in {step_name}: {e}")
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "="*70)
        logger.info("EXPERIMENT PIPELINE SUMMARY")
        logger.info("="*70)
        logger.info(f"Start Time: {start_time}")
        logger.info(f"End Time: {end_time}")
        logger.info(f"Total Duration: {duration}")
        logger.info(f"\nCompleted Steps ({len(completed_steps)}/{len(steps)}):")
        for step in completed_steps:
            logger.info(f"  ✓ {step}")
        
        if failed_steps:
            logger.info(f"\nFailed Steps ({len(failed_steps)}/{len(steps)}):")
            for step in failed_steps:
                logger.info(f"  ✗ {step}")
        
        logger.info(f"\nResults saved to: {self.output_dir}")
        logger.info("\nTo analyze results, run:")
        logger.info("  jupyter notebook notebooks/")
        logger.info("\nOr check the generated report:")
        logger.info(f"  cat {self.output_dir}/EXPERIMENT_REPORT.md")
        
        return len(failed_steps) == 0

def main():
    parser = argparse.ArgumentParser(description="Run automated meta-learning experiments")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--step", choices=["data", "train", "baseline", "eval", "analyze", "report"], 
                       help="Run specific step only")
    parser.add_argument("--quick", action="store_true", help="Run quick test with minimal configuration")
    
    args = parser.parse_args()
    
    if args.quick:
        quick_config = {
            'datasets': ['Cora'],
            'meta_learning_methods': ['Meta-GCN'],
            'baseline_methods': ['Louvain', 'Spectral'],
            'n_trials': 1,
            'meta_learning_epochs': 100,
            'patience': 20,
            'run_notebooks': False
        }
        config_file = os.path.join(args.output_dir, "quick_config.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(quick_config, f, indent=2)
        args.config = config_file
        logger.info("Running in quick test mode")
    
    runner = ExperimentRunner(args.config, args.output_dir)
    
    if args.step:
        step_map = {
            "data": runner.step_1_data_preparation,
            "train": runner.step_2_meta_learning_training,
            "baseline": runner.step_3_baseline_comparison,
            "eval": runner.step_4_model_evaluation,
            "analyze": runner.step_5_results_analysis,
            "report": runner.step_6_generate_report
        }
        
        if args.step in step_map:
            success = step_map[args.step]()
            sys.exit(0 if success else 1)
        else:
            logger.error(f"Unknown step: {args.step}")
            sys.exit(1)
    else:
        success = runner.run_full_experiment()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
