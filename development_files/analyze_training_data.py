"""
Training Data Analysis Script
Analyzes the training data export JSON file to improve rules and pattern detection
"""

import json
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import pandas as pd

def load_training_data(file_path: str) -> List[Dict]:
    """Load training data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ Loaded {len(data)} training records")
        return data
    except Exception as e:
        print(f"✗ Error loading training data: {e}")
        return []

def analyze_failure_codes(data: List[Dict]) -> Dict:
    """Analyze failure code distribution and patterns"""
    print("\n" + "="*60)
    print("FAILURE CODE ANALYSIS")
    print("="*60)
    
    failure_code_counts = Counter()
    failure_description_counts = Counter()
    code_to_descriptions = defaultdict(set)
    
    for record in data:
        code = record.get('assigned_code', 'Unknown')
        description = record.get('assigned_description', 'Unknown')
        work_desc = record.get('description', '')
        
        failure_code_counts[code] += 1
        failure_description_counts[description] += 1
        code_to_descriptions[code].add(description)
    
    print(f"Total unique failure codes: {len(failure_code_counts)}")
    print(f"Total unique failure descriptions: {len(failure_description_counts)}")
    
    print("\nTop 20 Failure Codes:")
    for code, count in failure_code_counts.most_common(20):
        descriptions = list(code_to_descriptions[code])[:3]  # Show first 3 descriptions
        print(f"  {code}: {count} occurrences - {descriptions}")
    
    return {
        'failure_code_counts': failure_code_counts,
        'failure_description_counts': failure_description_counts,
        'code_to_descriptions': code_to_descriptions
    }

def analyze_equipment_types(data: List[Dict]) -> Dict:
    """Analyze equipment type patterns"""
    print("\n" + "="*60)
    print("EQUIPMENT TYPE ANALYSIS")
    print("="*60)
    
    equipment_counts = Counter()
    equipment_failure_patterns = defaultdict(lambda: defaultdict(int))
    
    for record in data:
        spacy_analysis = record.get('spacy_analysis', {})
        equipment_types = spacy_analysis.get('equipment_types', [])
        failure_code = record.get('assigned_code', 'Unknown')
        work_desc = record.get('description', '')
        
        for equipment in equipment_types:
            equipment_counts[equipment] += 1
            equipment_failure_patterns[equipment][failure_code] += 1
    
    print(f"Total unique equipment types: {len(equipment_counts)}")
    
    print("\nTop 15 Equipment Types:")
    for equipment, count in equipment_counts.most_common(15):
        print(f"  {equipment}: {count} occurrences")
        
        # Show top failures for this equipment
        top_failures = sorted(equipment_failure_patterns[equipment].items(), 
                            key=lambda x: x[1], reverse=True)[:5]
        if top_failures:
            print(f"    Top failures: {dict(top_failures)}")
    
    return {
        'equipment_counts': equipment_counts,
        'equipment_failure_patterns': equipment_failure_patterns
    }

def analyze_failure_indicators(data: List[Dict]) -> Dict:
    """Analyze failure indicator patterns"""
    print("\n" + "="*60)
    print("FAILURE INDICATOR ANALYSIS")
    print("="*60)
    
    indicator_counts = Counter()
    indicator_failure_patterns = defaultdict(lambda: defaultdict(int))
    
    for record in data:
        spacy_analysis = record.get('spacy_analysis', {})
        failure_indicators = spacy_analysis.get('failure_indicators', [])
        failure_code = record.get('assigned_code', 'Unknown')
        
        for indicator in failure_indicators:
            indicator_counts[indicator] += 1
            indicator_failure_patterns[indicator][failure_code] += 1
    
    print(f"Total unique failure indicators: {len(indicator_counts)}")
    
    print("\nTop 20 Failure Indicators:")
    for indicator, count in indicator_counts.most_common(20):
        print(f"  {indicator}: {count} occurrences")
        
        # Show top failure codes for this indicator
        top_failures = sorted(indicator_failure_patterns[indicator].items(), 
                            key=lambda x: x[1], reverse=True)[:3]
        if top_failures:
            print(f"    Associated failures: {dict(top_failures)}")
    
    return {
        'indicator_counts': indicator_counts,
        'indicator_failure_patterns': indicator_failure_patterns
    }

def analyze_technical_terms(data: List[Dict]) -> Dict:
    """Analyze technical term patterns"""
    print("\n" + "="*60)
    print("TECHNICAL TERM ANALYSIS")
    print("="*60)
    
    technical_term_counts = Counter()
    term_failure_patterns = defaultdict(lambda: defaultdict(int))
    
    for record in data:
        spacy_analysis = record.get('spacy_analysis', {})
        technical_terms = spacy_analysis.get('technical_terms', [])
        failure_code = record.get('assigned_code', 'Unknown')
        
        for term in technical_terms:
            technical_term_counts[term] += 1
            term_failure_patterns[term][failure_code] += 1
    
    print(f"Total unique technical terms: {len(technical_term_counts)}")
    
    print("\nTop 30 Technical Terms:")
    for term, count in technical_term_counts.most_common(30):
        print(f"  {term}: {count} occurrences")
        
        # Show top failure codes for this term
        top_failures = sorted(term_failure_patterns[term].items(), 
                            key=lambda x: x[1], reverse=True)[:3]
        if top_failures:
            print(f"    Associated failures: {dict(top_failures)}")
    
    return {
        'technical_term_counts': technical_term_counts,
        'term_failure_patterns': term_failure_patterns
    }

def analyze_description_patterns(data: List[Dict]) -> Dict:
    """Analyze work order description patterns"""
    print("\n" + "="*60)
    print("DESCRIPTION PATTERN ANALYSIS")
    print("="*60)
    
    # Common phrases and patterns
    common_phrases = Counter()
    word_patterns = Counter()
    
    for record in data:
        description = record.get('description', '').lower()
        failure_code = record.get('assigned_code', 'Unknown')
        
        # Extract common phrases (2-4 words)
        words = description.split()
        for i in range(len(words) - 1):
            for j in range(i + 2, min(i + 5, len(words) + 1)):
                phrase = ' '.join(words[i:j])
                if len(phrase) > 3:  # Only meaningful phrases
                    common_phrases[phrase] += 1
        
        # Extract word patterns
        for word in words:
            if len(word) > 2:  # Skip very short words
                word_patterns[word] += 1
    
    print(f"Total unique phrases: {len(common_phrases)}")
    print(f"Total unique words: {len(word_patterns)}")
    
    print("\nTop 20 Common Phrases:")
    for phrase, count in common_phrases.most_common(20):
        print(f"  '{phrase}': {count} occurrences")
    
    print("\nTop 30 Common Words:")
    for word, count in word_patterns.most_common(30):
        print(f"  {word}: {count} occurrences")
    
    return {
        'common_phrases': common_phrases,
        'word_patterns': word_patterns
    }

def generate_rule_improvements(data: List[Dict], analysis_results: Dict) -> Dict:
    """Generate suggestions for rule improvements"""
    print("\n" + "="*60)
    print("RULE IMPROVEMENT SUGGESTIONS")
    print("="*60)
    
    suggestions = {
        'new_expert_rules': [],
        'enhanced_patterns': [],
        'equipment_contexts': [],
        'keyword_additions': []
    }
    
    # Analyze patterns for new expert rules
    failure_code_counts = analysis_results['failure_code_counts']
    indicator_failure_patterns = analysis_results['indicator_failure_patterns']
    
    # Find failure codes that appear frequently but may not have expert rules
    common_failures = [code for code, count in failure_code_counts.most_common(30) 
                      if count > 5]  # More than 5 occurrences
    
    # Check for missing expert rules
    existing_rules = ['bearing_failure', 'seal_leak', 'motor_overheating', 'pump_cavitation', 
                     'electrical_fault', 'valve_stuck', 'belt_failure', 'corrosion',
                     'packing_leak', 'lubrication_failure', 'signal_fault']
    
    for failure in common_failures:
        if failure.lower() not in [rule.replace('_', ' ') for rule in existing_rules]:
            # Find associated indicators and terms
            associated_indicators = []
            for indicator, patterns in indicator_failure_patterns.items():
                if failure in patterns:
                    associated_indicators.append(indicator)
            
            if associated_indicators:
                suggestions['new_expert_rules'].append({
                    'failure_code': failure,
                    'indicators': associated_indicators[:5],  # Top 5 indicators
                    'frequency': failure_code_counts[failure]
                })
    
    # Generate enhanced patterns
    technical_term_counts = analysis_results['technical_term_counts']
    common_phrases = analysis_results['common_phrases']
    
    # Find technical terms that appear frequently
    frequent_terms = [term for term, count in technical_term_counts.most_common(50) 
                     if count > 3 and len(term) > 2]
    
    for term in frequent_terms:
        suggestions['enhanced_patterns'].append({
            'term': term,
            'frequency': technical_term_counts[term],
            'pattern': f"r'{re.escape(term)}'"
        })
    
    # Equipment context improvements
    equipment_counts = analysis_results['equipment_counts']
    equipment_failure_patterns = analysis_results['equipment_failure_patterns']
    
    for equipment, count in equipment_counts.most_common(20):
        if count > 2:  # Equipment with multiple occurrences
            top_failures = sorted(equipment_failure_patterns[equipment].items(), 
                                key=lambda x: x[1], reverse=True)[:5]
            
            suggestions['equipment_contexts'].append({
                'equipment': equipment,
                'frequency': count,
                'common_failures': dict(top_failures)
            })
    
    return suggestions

def print_rule_improvements(suggestions: Dict):
    """Print rule improvement suggestions"""
    print("\n" + "="*60)
    print("DETAILED RULE IMPROVEMENTS")
    print("="*60)
    
    print("\n1. NEW EXPERT RULES SUGGESTIONS:")
    for rule in suggestions['new_expert_rules'][:10]:  # Top 10
        print(f"  Failure: {rule['failure_code']} ({rule['frequency']} occurrences)")
        print(f"    Indicators: {rule['indicators']}")
        print()
    
    print("\n2. ENHANCED PATTERN SUGGESTIONS:")
    for pattern in suggestions['enhanced_patterns'][:15]:  # Top 15
        print(f"  Term: '{pattern['term']}' ({pattern['frequency']} occurrences)")
        print(f"    Pattern: {pattern['pattern']}")
        print()
    
    print("\n3. EQUIPMENT CONTEXT IMPROVEMENTS:")
    for context in suggestions['equipment_contexts'][:10]:  # Top 10
        print(f"  Equipment: {context['equipment']} ({context['frequency']} occurrences)")
        print(f"    Common failures: {context['common_failures']}")
        print()

def generate_code_improvements(suggestions: Dict) -> str:
    """Generate code improvements for the AI classifier"""
    print("\n" + "="*60)
    print("CODE IMPROVEMENTS")
    print("="*60)
    
    code_improvements = []
    
    # Generate new expert rules
    if suggestions['new_expert_rules']:
        code_improvements.append("# New Expert Rules to Add:")
        for rule in suggestions['new_expert_rules'][:5]:  # Top 5
            code_improvements.append(f"""        {{
            'name': '{rule['failure_code'].lower().replace(' ', '_')}',
            'conditions': [""")
            
            for indicator in rule['indicators'][:3]:  # Top 3 indicators
                code_improvements.append(f"                {{'type': 'keyword', 'value': '{indicator}', 'weight': 0.3}},")
            
            code_improvements.append(f"""            ],
            'failure_code': '{rule['failure_code']}',
            'description': '{rule['failure_code']}'
        }},""")
    
    # Generate enhanced patterns
    if suggestions['enhanced_patterns']:
        code_improvements.append("\n# Enhanced Patterns to Add:")
        for pattern in suggestions['enhanced_patterns'][:10]:  # Top 10
            code_improvements.append(f"    '{pattern['term']}': {pattern['pattern']},")
    
    # Generate equipment context improvements
    if suggestions['equipment_contexts']:
        code_improvements.append("\n# Equipment Context Improvements:")
        for context in suggestions['equipment_contexts'][:5]:  # Top 5
            code_improvements.append(f"""            '{context['equipment']}': {{
                'common_failures': {list(context['common_failures'].keys())},
                'context_words': ['{context['equipment']}'],
                'failure_patterns': {{
                    # Add patterns based on common failures
                }}
            }},""")
    
    return '\n'.join(code_improvements)

def main():
    """Main analysis function"""
    print("Training Data Analysis for Rule Improvement")
    print("="*60)
    
    # Load training data
    data = load_training_data('training_data_export.json')
    if not data:
        return
    
    # Perform analysis
    analysis_results = {}
    analysis_results.update(analyze_failure_codes(data))
    analysis_results.update(analyze_equipment_types(data))
    analysis_results.update(analyze_failure_indicators(data))
    analysis_results.update(analyze_technical_terms(data))
    analysis_results.update(analyze_description_patterns(data))
    
    # Generate improvements
    suggestions = generate_rule_improvements(data, analysis_results)
    print_rule_improvements(suggestions)
    
    # Generate code improvements
    code_improvements = generate_code_improvements(suggestions)
    
    # Save analysis results
    with open('training_analysis_results.json', 'w') as f:
        json.dump({
            'analysis_results': {k: dict(v) if isinstance(v, Counter) else v 
                               for k, v in analysis_results.items()},
            'suggestions': suggestions,
            'code_improvements': code_improvements
        }, f, indent=2, default=str)
    
    print(f"\n✓ Analysis complete! Results saved to 'training_analysis_results.json'")
    print(f"✓ Found {len(suggestions['new_expert_rules'])} potential new expert rules")
    print(f"✓ Found {len(suggestions['enhanced_patterns'])} enhanced patterns")
    print(f"✓ Found {len(suggestions['equipment_contexts'])} equipment context improvements")

if __name__ == "__main__":
    main() 