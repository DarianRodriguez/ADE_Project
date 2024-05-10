
import torch
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def extract_entity_span(num_labels,labels):
    unique_labels = list(range(1, num_labels))
    entities = []

    for i in range(0, len(unique_labels), 2):
        pair = unique_labels[i:i+2]

        # Find indices where transition from 1 to 2 occurs (Drug)
        #transition_1_to_2_indices = torch.where((labels[:-1] == pair[0]) & (labels[1:] == pair[1]))[0] 
        transition_1_to_2_indices = torch.where((labels[:-1] == pair[0]) & (labels[1:] == pair[1]))[0] 
        transition_2_to_any = torch.where((labels[:-1] == pair[1]) & (labels[1:] != pair[1]))[0]

        if len(transition_2_to_any) != len(transition_1_to_2_indices): 
            length = len(labels)-1
            transition_2_to_any = torch.cat((transition_2_to_any, torch.tensor([length], device=labels.device)))

        entity = zip(transition_1_to_2_indices,transition_2_to_any)
        entities.append(entity)
    
    
    return entities

def count_instances(labels, predictions,transition):
    """ Count Spurius (SPU) """
    # Convert tensors to NumPy arrays
    labels_np = labels.cpu().numpy()
    predictions_np = predictions.cpu().numpy()

    # Identify the indices where the labels have 0
    zero_indices = np.where(labels_np == 0)[0]

    # Identify the indices where the predictions have a transition from 1 to 2
    transition_indices = np.where((predictions_np[:-1] == transition[0]) & (predictions_np[1:] == transition[1]))[0] + 1

    # Perform intersection using NumPy
    overlapping_indices_np = np.intersect1d(zero_indices, transition_indices)

    # Convert the result back to a PyTorch tensor
    overlapping_indices = torch.from_numpy(overlapping_indices_np)

    # Count the instances where the labels have 0 at the overlapping indices
    count = torch.sum(labels[overlapping_indices] == 0)

    return count.item()


def calculate_metrics(entity, true_labels, predicted_labels,entity_change = [1,2]):

    TP, FP, FN = 0, 0, 0

    for start, end in entity:
        true_span = true_labels[start:end].cpu().numpy()
        predicted_span = predicted_labels[start:end].cpu().numpy()

        # Check if the spans are equal
        if np.array_equal(true_span, predicted_span):
            TP += 1
        else:
            FN += 1

    FP= count_instances(true_labels, predicted_labels,entity_change)
    #print("FP",FP)

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    # Accuracy calculation
    accuracy = (TP) / (TP + FP + FN) if TP + FP + FN > 0 else 0

    # Print the calculated metrics
    print("Precision: {:.3f}".format(precision))
    print("Recall: {:.3f}".format(recall))
    print("F1-score: {:.3f}".format(f1_score))
    print("Accuracy: {:.3f}".format(accuracy))


    #print("TP",TP)

    return

def evaluate(true_labels,eval_predictions):

    non_padding_mask = true_labels != -100

    # Apply the mask to both true labels and predictions
    masked_true_labels = true_labels[non_padding_mask]
    masked_predictions = eval_predictions[non_padding_mask]

    entities = extract_entity_span(5,masked_true_labels)

    print("####### DRUGS #######")
    calculate_metrics(entities[0], masked_true_labels, masked_predictions,[1,2]) # Drugs

    print("####### EFFECTS #######")
    calculate_metrics(entities[1], masked_true_labels, masked_predictions,[3,4]) # Effect

    label_mapping = {
    0: 'O',
    1: 'B-drug',
    2: 'I-drug',
    3: 'B-effect',
    4: 'I-effect'
    }

    labels = list(label_mapping.values())

    # Convert tensors to numpy arrays
    masked_true_labels_np = masked_true_labels.cpu().numpy()
    masked_predictions_np = masked_predictions.cpu().numpy()

    # Generate classification report
    report = classification_report(masked_true_labels_np, masked_predictions_np,target_names=labels)

    # Compute precision, recall, and F1 score
    precision, recall, f1_score, _ = precision_recall_fscore_support(masked_true_labels_np, masked_predictions_np, average='weighted')

    print("####### Classification Report: #######")
    print(report)

    print("Precision: {:.3f}".format(precision))
    print("Recall: {:.3f}".format(recall))
    print("F1 Score: {:.3f}".format(f1_score))

    return


def generate_classification_report_re(threshold, predictions, true_labels, target_names,attention_masks):

    # Applying threshold using torch.where
    predictions_new = torch.where(predictions > threshold, torch.tensor(1), torch.tensor(0))
    # Apply attention mask to filter out predictions and true labels
    predictions_masked = predictions_new[attention_masks.bool()]
    true_labels_masked = true_labels[attention_masks.bool()]

    # Convert predictions and true labels to numpy arrays
    masked_true_labels_np = true_labels_masked.cpu().numpy().ravel()
    masked_predictions_np = predictions_masked.cpu().numpy().ravel()

    # Generate classification report
    report = classification_report(masked_true_labels_np, masked_predictions_np, target_names=target_names)

    # Compute precision, recall, and F1 score
    precision, recall, f1_score, _ = precision_recall_fscore_support(masked_true_labels_np, masked_predictions_np, average='weighted')

    print("####### Classification Report: #######")
    print(report)

    print("Precision: {:.3f}".format(precision))
    print("Recall: {:.3f}".format(recall))
    print("F1 Score: {:.3f}".format(f1_score))

    return