def evaluate_forgetting(model, baseline_data, current_data):
    """Evaluate backward transfer (forgetting) as in the paper"""
    
    def compute_accuracy(model, data):
        correct = 0
        total = len(data)
        
        for item in data:
            # Generate response and check accuracy
            response = generate_response(model, item['text'])
            if evaluate_response_quality(response, item['label']):
                correct += 1
        
        return correct / total
    
    # Compute metrics from paper
    baseline_acc = compute_accuracy(model, baseline_data)
    current_acc = compute_accuracy(model, current_data)
    
    # Backward Transfer (BWT) - negative values indicate forgetting
    bwt = baseline_acc - original_baseline_acc
    
    return {
        "baseline_accuracy": baseline_acc,
        "current_accuracy": current_acc,
        "backward_transfer": bwt,
        "forgetting_acceptable": bwt > -0.15  # 15% threshold from paper
    }