import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict
from train_aspect_aware import Logger


def plot_training_curves(
    train_losses: List[float], 
    val_losses: List[float], 
    val_accuracies: List[float], 
    save_path: Path,
    logger: Logger
):
    """Plot training curves"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Total loss curve
        ax1.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
        ax1.plot(val_losses, label='Val Loss', color='red', linewidth=2)
        ax1.set_title('Train & Val Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curve
        ax2.plot(val_accuracies, label='Val Accuracy', color='green', linewidth=2)
        ax2.set_title('Val Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Training curves saved to: {save_path / 'training_curves.png'}")
    except Exception as e:
        logger.error(f"Failed to plot training curves: {e}")


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str] = None,
    save_path: Path = None,
    logger: Logger = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    figsize: tuple = (10, 8),
    cmap: str = 'Blues',
    show_values: bool = True,
    fontsize: int = 12
):
    """
    Plot confusion matrix
    
    Args:
        y_true: List of true labels
        y_pred: List of predicted labels
        class_names: List of class names, if None use numeric labels
        save_path: Save path, if None only display without saving
        logger: Logger for recording save information
        normalize: Whether to normalize confusion matrix
        title: Chart title
        figsize: Chart size
        cmap: Color map
        show_values: Whether to show values in matrix
        fontsize: Font size
    """
    try:
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm)  # Handle division by zero
        
        # Set class names
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(cm))]
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot confusion matrix using seaborn
        sns.heatmap(
            cm, 
            annot=show_values, 
            fmt='.2f' if normalize else 'd',
            cmap=cmap,
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True,
            square=True,
            linewidths=0.5,
            linecolor='white'
        )
        
        # Set title and labels
        plt.title(title, fontsize=fontsize + 2, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=fontsize)
        plt.ylabel('True Label', fontsize=fontsize)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save image
        if save_path is not None:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
            if logger is not None:
                logger.info(f"Confusion matrix saved to: {save_path / 'confusion_matrix.png'}")
        
        # Calculate and return evaluation metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        if logger is not None:
            logger.info(f"Model Evaluation Metrics:")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            logger.info(f"  F1 Score: {f1:.4f}")
        
        return metrics
        
    except ImportError as e:
        error_msg = f"Missing required dependencies: {e}"
        if logger is not None:
            logger.error(error_msg)
        else:
            print(f"Error: {error_msg}")
        return None
        
    except Exception as e:
        error_msg = f"Failed to plot confusion matrix: {e}"
        if logger is not None:
            logger.error(error_msg)
        else:
            print(f"Error: {error_msg}")
        return None


def plot_classification_report(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str] = None,
    save_path: Path = None,
    logger: Logger = None,
    title: str = "Classification Report",
    figsize: tuple = (12, 8)
):
    """
    Plot classification report heatmap
    
    Args:
        y_true: List of true labels
        y_pred: List of predicted labels
        class_names: List of class names
        save_path: Save path
        logger: Logger
        title: Chart title
        figsize: Chart size
    """
    try:
        from sklearn.metrics import classification_report
        import pandas as pd
        
        # Generate classification report
        report = classification_report(
            y_true, 
            y_pred, 
            target_names=class_names,
            output_dict=True
        )
        
        # Convert to DataFrame
        df_report = pd.DataFrame(report).transpose()
        
        # Remove last row (accuracy)
        df_report = df_report.iloc[:-1, :]
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(
            df_report, 
            annot=True, 
            fmt='.3f',
            cmap='YlOrRd',
            cbar=True,
            square=False,
            linewidths=0.5,
            linecolor='white'
        )
        
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Classes', fontsize=12)
        
        plt.tight_layout()
        
        # Save image
        if save_path is not None:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path / 'classification_report.png', dpi=300, bbox_inches='tight')
            if logger is not None:
                logger.info(f"Classification report saved to: {save_path / 'classification_report.png'}")
        
        
    except ImportError as e:
        error_msg = f"Missing required dependencies: {e}"
        if logger is not None:
            logger.error(error_msg)
        else:
            print(f"Error: {error_msg}")
            
    except Exception as e:
        error_msg = f"Failed to plot classification report: {e}"
        if logger is not None:
            logger.error(error_msg)
        else:
            print(f"Error: {error_msg}")