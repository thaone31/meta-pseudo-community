#!/usr/bin/env python3
"""
Summary Report Generator for Meta-Learning Pseudo-Labels Community Detection

This utility generates comprehensive summary reports from experimental results.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import glob

class SummaryGenerator:
    def __init__(self, results_dir, output_file=None):
        self.results_dir = Path(results_dir)
        self.output_file = output_file or self.results_dir / "summary_report.json"
        
    def collect_all_results(self):
        """Collect all experimental results from the results directory."""
        results = {
            'meta_learning': [],
            'baselines': [],
            'training_logs': [],
            'evaluation_metrics': [],
            'dataset_stats': []
        }
        
        # Collect JSON result files
        json_files = list(self.results_dir.rglob("*.json"))
        
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                file_name = file_path.name.lower()
                
                if 'meta' in file_name or 'maml' in file_name or 'reptile' in file_name:
                    results['meta_learning'].append({
                        'file': str(file_path),
                        'data': data
                    })
                elif 'baseline' in file_name or 'comparison' in file_name:
                    results['baselines'].append({
                        'file': str(file_path),
                        'data': data
                    })
                elif 'training' in file_name:
                    results['training_logs'].append({
                        'file': str(file_path),
                        'data': data
                    })
                elif 'evaluation' in file_name:
                    results['evaluation_metrics'].append({
                        'file': str(file_path),
                        'data': data
                    })
                elif 'dataset' in file_name:
                    results['dataset_stats'].append({
                        'file': str(file_path),
                        'data': data
                    })
                    
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
        
        # Collect CSV files
        csv_files = list(self.results_dir.rglob("*.csv"))
        
        csv_data = {}
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                csv_data[file_path.name] = {
                    'file': str(file_path),
                    'shape': df.shape,
                    'columns': df.columns.tolist(),
                    'summary': df.describe().to_dict() if not df.empty else {}
                }
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
        
        results['csv_data'] = csv_data
        
        return results
    
    def analyze_performance_results(self, results):
        """Analyze performance across all methods and datasets."""
        all_results = []
        
        # Extract results from meta-learning experiments
        for result in results['meta_learning']:
            data = result['data']
            if isinstance(data, dict):
                if 'results' in data:
                    if isinstance(data['results'], list):
                        all_results.extend(data['results'])
                    elif isinstance(data['results'], dict):
                        all_results.append(data['results'])
                else:
                    all_results.append(data)
        
        # Extract results from baseline experiments
        for result in results['baselines']:
            data = result['data']
            if isinstance(data, dict):
                if 'results' in data:
                    if isinstance(data['results'], list):
                        all_results.extend(data['results'])
                    elif isinstance(data['results'], dict):
                        all_results.append(data['results'])
                else:
                    all_results.append(data)
        
        if not all_results:\n            return {}\n        \n        # Convert to DataFrame for analysis\n        try:\n            df = pd.DataFrame(all_results)\n        except:\n            # If direct conversion fails, try to normalize the data\n            normalized_results = []\n            for result in all_results:\n                if isinstance(result, dict):\n                    normalized_results.append(result)\n                else:\n                    normalized_results.append({'result': result})\n            df = pd.DataFrame(normalized_results)\n        \n        if df.empty:\n            return {}\n        \n        performance_analysis = {\n            'total_experiments': len(df),\n            'unique_methods': df['method'].nunique() if 'method' in df.columns else 0,\n            'unique_datasets': df['dataset'].nunique() if 'dataset' in df.columns else 0,\n            'metrics_analyzed': []\n        }\n        \n        # Analyze each metric\n        metric_columns = ['NMI', 'ARI', 'Modularity', 'Conductance', 'F1', 'Precision', 'Recall']\n        available_metrics = [col for col in metric_columns if col in df.columns]\n        \n        for metric in available_metrics:\n            metric_stats = {\n                'metric': metric,\n                'count': df[metric].count(),\n                'mean': float(df[metric].mean()),\n                'std': float(df[metric].std()),\n                'min': float(df[metric].min()),\n                'max': float(df[metric].max()),\n                'median': float(df[metric].median())\n            }\n            \n            # Best performing method for this metric\n            if 'method' in df.columns:\n                method_performance = df.groupby('method')[metric].mean().sort_values(ascending=False)\n                if not method_performance.empty:\n                    metric_stats['best_method'] = method_performance.index[0]\n                    metric_stats['best_score'] = float(method_performance.iloc[0])\n            \n            performance_analysis['metrics_analyzed'].append(metric_stats)\n        \n        # Method comparison\n        if 'method' in df.columns and available_metrics:\n            method_comparison = {}\n            for method in df['method'].unique():\n                method_data = df[df['method'] == method]\n                method_stats = {}\n                for metric in available_metrics:\n                    if metric in method_data.columns:\n                        method_stats[metric] = {\n                            'mean': float(method_data[metric].mean()),\n                            'std': float(method_data[metric].std()),\n                            'count': int(method_data[metric].count())\n                        }\n                method_comparison[method] = method_stats\n            \n            performance_analysis['method_comparison'] = method_comparison\n        \n        # Dataset analysis\n        if 'dataset' in df.columns and available_metrics:\n            dataset_analysis = {}\n            for dataset in df['dataset'].unique():\n                dataset_data = df[df['dataset'] == dataset]\n                dataset_stats = {}\n                for metric in available_metrics:\n                    if metric in dataset_data.columns:\n                        dataset_stats[metric] = {\n                            'mean': float(dataset_data[metric].mean()),\n                            'std': float(dataset_data[metric].std()),\n                            'count': int(dataset_data[metric].count())\n                        }\n                dataset_analysis[dataset] = dataset_stats\n            \n            performance_analysis['dataset_analysis'] = dataset_analysis\n        \n        return performance_analysis\n    \n    def analyze_training_progress(self, results):\n        \"\"\"Analyze training convergence and progress.\"\"\"\n        training_analysis = {\n            'total_training_runs': len(results['training_logs']),\n            'convergence_analysis': [],\n            'training_summary': {}\n        }\n        \n        all_training_data = []\n        \n        for log in results['training_logs']:\n            data = log['data']\n            if isinstance(data, dict) and 'history' in data:\n                history = data['history']\n                if isinstance(history, list) and history:\n                    # Analyze convergence\n                    df = pd.DataFrame(history)\n                    \n                    convergence_info = {\n                        'file': log['file'],\n                        'total_iterations': len(df),\n                        'final_loss': None,\n                        'loss_reduction': None,\n                        'converged': False\n                    }\n                    \n                    if 'meta_loss' in df.columns:\n                        final_loss = df['meta_loss'].iloc[-1]\n                        initial_loss = df['meta_loss'].iloc[0]\n                        loss_reduction = (initial_loss - final_loss) / initial_loss\n                        \n                        convergence_info.update({\n                            'final_loss': float(final_loss),\n                            'loss_reduction': float(loss_reduction),\n                            'converged': loss_reduction > 0.5  # Arbitrary threshold\n                        })\n                    \n                    training_analysis['convergence_analysis'].append(convergence_info)\n                    all_training_data.extend(history)\n        \n        # Overall training summary\n        if all_training_data:\n            df_all = pd.DataFrame(all_training_data)\n            summary = {}\n            \n            for col in ['meta_loss', 'adaptation_loss', 'pseudo_label_nmi', 'pseudo_label_ari']:\n                if col in df_all.columns:\n                    summary[col] = {\n                        'mean': float(df_all[col].mean()),\n                        'std': float(df_all[col].std()),\n                        'final_avg': float(df_all[col].tail(100).mean()) if len(df_all) > 100 else float(df_all[col].mean())\n                    }\n            \n            training_analysis['training_summary'] = summary\n        \n        return training_analysis\n    \n    def generate_summary_statistics(self, results):\n        \"\"\"Generate overall summary statistics.\"\"\"\n        summary_stats = {\n            'generation_time': datetime.now().isoformat(),\n            'results_directory': str(self.results_dir),\n            'file_counts': {\n                'json_files': len(results['meta_learning']) + len(results['baselines']) + \n                             len(results['training_logs']) + len(results['evaluation_metrics']) + \n                             len(results['dataset_stats']),\n                'csv_files': len(results['csv_data'])\n            },\n            'experiment_summary': {\n                'meta_learning_experiments': len(results['meta_learning']),\n                'baseline_experiments': len(results['baselines']),\n                'training_logs': len(results['training_logs']),\n                'evaluation_files': len(results['evaluation_metrics']),\n                'dataset_analysis_files': len(results['dataset_stats'])\n            }\n        }\n        \n        # Check for specific result types\n        has_performance_results = any([\n            results['meta_learning'],\n            results['baselines'],\n            results['evaluation_metrics']\n        ])\n        \n        has_training_results = bool(results['training_logs'])\n        \n        summary_stats['analysis_capabilities'] = {\n            'can_analyze_performance': has_performance_results,\n            'can_analyze_training': has_training_results,\n            'can_compare_methods': has_performance_results,\n            'has_dataset_statistics': bool(results['dataset_stats'])\n        }\n        \n        return summary_stats\n    \n    def generate_recommendations(self, performance_analysis, training_analysis):\n        \"\"\"Generate recommendations based on the analysis.\"\"\"\n        recommendations = []\n        \n        # Performance recommendations\n        if performance_analysis:\n            total_experiments = performance_analysis.get('total_experiments', 0)\n            \n            if total_experiments == 0:\n                recommendations.append({\n                    'type': 'critical',\n                    'message': 'No performance results found. Run experiments first.'\n                })\n            elif total_experiments < 10:\n                recommendations.append({\n                    'type': 'warning',\n                    'message': f'Only {total_experiments} experiments found. Consider running more trials for statistical significance.'\n                })\n            \n            # Method performance recommendations\n            if 'metrics_analyzed' in performance_analysis:\n                for metric_info in performance_analysis['metrics_analyzed']:\n                    if metric_info['std'] > metric_info['mean'] * 0.3:\n                        recommendations.append({\n                            'type': 'warning',\n                            'message': f\"High variance in {metric_info['metric']} scores. Consider more stable methods or parameter tuning.\"\n                        })\n        \n        # Training recommendations\n        if training_analysis:\n            convergence_info = training_analysis.get('convergence_analysis', [])\n            \n            if convergence_info:\n                converged_runs = sum(1 for run in convergence_info if run.get('converged', False))\n                total_runs = len(convergence_info)\n                \n                if converged_runs < total_runs * 0.5:\n                    recommendations.append({\n                        'type': 'warning',\n                        'message': f'Only {converged_runs}/{total_runs} training runs converged. Consider adjusting learning rates or training longer.'\n                    })\n                \n                # Check for very long training\n                avg_iterations = np.mean([run.get('total_iterations', 0) for run in convergence_info])\n                if avg_iterations > 5000:\n                    recommendations.append({\n                        'type': 'info',\n                        'message': f'Training runs are long (avg {avg_iterations:.0f} iterations). Consider early stopping or faster convergence methods.'\n                    })\n        \n        # General recommendations\n        if not performance_analysis and not training_analysis:\n            recommendations.append({\n                'type': 'info',\n                'message': 'No detailed analysis possible with current results. Run full experimental pipeline.'\n            })\n        else:\n            recommendations.append({\n                'type': 'success',\n                'message': 'Analysis completed successfully. Review detailed results in Jupyter notebooks.'\n            })\n        \n        return recommendations\n    \n    def generate_report(self):\n        \"\"\"Generate comprehensive summary report.\"\"\"\n        print(f\"Generating summary report from {self.results_dir}...\")\n        \n        # Collect all results\n        results = self.collect_all_results()\n        \n        # Perform analyses\n        performance_analysis = self.analyze_performance_results(results)\n        training_analysis = self.analyze_training_progress(results)\n        summary_stats = self.generate_summary_statistics(results)\n        recommendations = self.generate_recommendations(performance_analysis, training_analysis)\n        \n        # Compile final report\n        report = {\n            'summary_statistics': summary_stats,\n            'performance_analysis': performance_analysis,\n            'training_analysis': training_analysis,\n            'recommendations': recommendations,\n            'raw_file_info': {\n                'meta_learning_files': [r['file'] for r in results['meta_learning']],\n                'baseline_files': [r['file'] for r in results['baselines']],\n                'training_files': [r['file'] for r in results['training_logs']],\n                'csv_files': list(results['csv_data'].keys())\n            }\n        }\n        \n        # Save report\n        with open(self.output_file, 'w') as f:\n            json.dump(report, f, indent=2)\n        \n        print(f\"Summary report saved to: {self.output_file}\")\n        \n        # Print summary to console\n        self.print_summary(report)\n        \n        return report\n    \n    def print_summary(self, report):\n        \"\"\"Print a human-readable summary to console.\"\"\"\n        print(\"\\n\" + \"=\"*60)\n        print(\"EXPERIMENTAL RESULTS SUMMARY\")\n        print(\"=\"*60)\n        \n        summary_stats = report['summary_statistics']\n        print(f\"Results Directory: {summary_stats['results_directory']}\")\n        print(f\"Generated: {summary_stats['generation_time']}\")\n        \n        print(f\"\\nFiles Found:\")\n        exp_summary = summary_stats['experiment_summary']\n        print(f\"  Meta-learning experiments: {exp_summary['meta_learning_experiments']}\")\n        print(f\"  Baseline experiments: {exp_summary['baseline_experiments']}\")\n        print(f\"  Training logs: {exp_summary['training_logs']}\")\n        print(f\"  Evaluation files: {exp_summary['evaluation_files']}\")\n        print(f\"  Dataset analysis files: {exp_summary['dataset_analysis_files']}\")\n        print(f\"  CSV data files: {summary_stats['file_counts']['csv_files']}\")\n        \n        # Performance summary\n        perf_analysis = report['performance_analysis']\n        if perf_analysis:\n            print(f\"\\nPerformance Analysis:\")\n            print(f\"  Total experiments: {perf_analysis.get('total_experiments', 0)}\")\n            print(f\"  Unique methods: {perf_analysis.get('unique_methods', 0)}\")\n            print(f\"  Unique datasets: {perf_analysis.get('unique_datasets', 0)}\")\n            \n            if 'metrics_analyzed' in perf_analysis:\n                print(f\"  Metrics analyzed: {len(perf_analysis['metrics_analyzed'])}\")\n                for metric_info in perf_analysis['metrics_analyzed']:\n                    print(f\"    {metric_info['metric']}: {metric_info['mean']:.3f} ± {metric_info['std']:.3f}\")\n        \n        # Training summary\n        training_analysis = report['training_analysis']\n        if training_analysis and training_analysis['total_training_runs'] > 0:\n            print(f\"\\nTraining Analysis:\")\n            print(f\"  Training runs: {training_analysis['total_training_runs']}\")\n            \n            convergence_info = training_analysis.get('convergence_analysis', [])\n            if convergence_info:\n                converged = sum(1 for run in convergence_info if run.get('converged', False))\n                print(f\"  Converged runs: {converged}/{len(convergence_info)}\")\n        \n        # Recommendations\n        recommendations = report['recommendations']\n        if recommendations:\n            print(f\"\\nRecommendations:\")\n            for rec in recommendations:\n                icon = {\"critical\": \"⚠️\", \"warning\": \"⚠️\", \"info\": \"ℹ️\", \"success\": \"✅\"}.get(rec['type'], \"•\")\n                print(f\"  {icon} {rec['message']}\")\n        \n        print(\"\\n\" + \"=\"*60)\n\ndef main():\n    parser = argparse.ArgumentParser(description=\"Generate summary report from experimental results\")\n    parser.add_argument(\"--results-dir\", required=True, help=\"Directory containing experimental results\")\n    parser.add_argument(\"--output\", help=\"Output file for summary report\")\n    parser.add_argument(\"--print-only\", action=\"store_true\", help=\"Only print summary, don't save file\")\n    \n    args = parser.parse_args()\n    \n    if not os.path.exists(args.results_dir):\n        print(f\"Error: Results directory {args.results_dir} does not exist\")\n        sys.exit(1)\n    \n    output_file = args.output\n    if not output_file:\n        output_file = os.path.join(args.results_dir, \"summary_report.json\")\n    \n    generator = SummaryGenerator(args.results_dir, output_file)\n    \n    try:\n        report = generator.generate_report()\n        \n        if not args.print_only:\n            print(f\"\\nDetailed report saved to: {output_file}\")\n            print(\"\\nTo view the full report:\")\n            print(f\"  cat {output_file} | jq .\")\n            print(\"\\nTo analyze results interactively:\")\n            print(\"  jupyter notebook notebooks/\")\n        \n    except Exception as e:\n        print(f\"Error generating summary report: {e}\")\n        import traceback\n        traceback.print_exc()\n        sys.exit(1)\n\nif __name__ == \"__main__\":\n    main()
