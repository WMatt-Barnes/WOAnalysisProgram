"""
Debug script to test expert system scoring
"""

from ai_failure_classifier import ExpertSystemClassifier

def test_expert_system():
    """Test expert system scoring"""
    print("Testing Expert System Scoring")
    print("="*40)
    
    expert = ExpertSystemClassifier()
    
    test_cases = [
        'Aux Boiler TDL showing malfunction',
        '010PSV009B is leaking by and tubing fittings have a small drip',
        'I/A Dryer condensate trap not functioning properly',
        'Troubleshoot oil analyzer on NT-1700B',
        'Valve in tracking and not meeting limit switch',
        'HP-MP Letdown 015PV1501C will need to be calibrated',
        'Reconnect power to wall louvers',
        'Replace plastic chains at ECR transformers',
        'Seal leak on pump',
        'Motor overheating'
    ]
    
    for description in test_cases:
        print(f"\nDescription: {description}")
        code, desc, confidence = expert.classify(description)
        print(f"Result: {code} - {desc} (confidence: {confidence:.3f})")
        
        # Show detailed scoring
        description_lower = description.lower()
        for rule in expert.rules:
            score = 0.0
            matched_conditions = 0
            
            for condition in rule['conditions']:
                if condition['type'] == 'keyword':
                    if condition['value'].lower() in description_lower:
                        score += condition['weight']
                        matched_conditions += 1
                        print(f"  ✓ Keyword match: '{condition['value']}' (weight: {condition['weight']})")
                elif condition['type'] == 'pattern':
                    if re.search(condition['value'], description_lower, re.IGNORECASE):
                        score += condition['weight']
                        matched_conditions += 1
                        print(f"  ✓ Pattern match: '{condition['value']}' (weight: {condition['weight']})")
            
            if matched_conditions > 0:
                normalized_score = min(1.0, score / len(rule['conditions']))
                print(f"  Rule '{rule['name']}': raw_score={score:.3f}, matched={matched_conditions}, normalized={normalized_score:.3f}")

if __name__ == "__main__":
    import re
    test_expert_system() 