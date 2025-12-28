def confidence_label(score):
    if score >= 85:
        return "High Confidence ", "green"
    elif score >= 75:
        return "Medium Confidence ", "orange"
    else:
        return "Low Confidence ", "red"