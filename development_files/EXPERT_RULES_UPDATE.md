# Expert Rules Update - New Failure Types

## Overview
Added three new expert rules to the AI failure classifier to enhance failure mode classification capabilities:

1. **Packing Leak** (Code: 9.1)
2. **Lubrication Failure** (Code: 10.1) 
3. **Signal Fault/Indication Error** (Code: 11.1)

## New Expert Rules Details

### 1. Packing Leak (Code: 9.1)
**Description**: Packing Leak

**Keywords and Patterns**:
- Keywords: `packing` (0.4), `leak` (0.3), `gland` (0.3)
- Patterns:
  - `packing.*leak` (0.5)
  - `gland.*leak` (0.4)
  - `packing.*seal` (0.3)
  - `stuffing.*box` (0.3)

**Differentiation**: Specifically designed to differentiate from mechanical seal leaks (Code: 2.1)

### 2. Lubrication Failure (Code: 10.1)
**Description**: Lubrication Failure

**Keywords and Patterns**:
- Keywords: `lubrication` (0.4), `lubricant` (0.3), `oil` (0.2), `grease` (0.2), `dry` (0.3)
- Patterns:
  - `lubrication.*fail` (0.5)
  - `oil.*level.*low` (0.4)
  - `grease.*not.*flow` (0.4)
  - `dry.*running` (0.4)
  - `no.*lubrication` (0.5)

**Coverage**: Covers oil, grease, and dry running conditions

### 3. Signal Fault/Indication Error (Code: 11.1)
**Description**: Signal Fault/Indication Error

**Keywords and Patterns**:
- Keywords: `signal` (0.4), `indication` (0.3), `alarm` (0.3), `reading` (0.2), `display` (0.2), `sensor` (0.2)
- Patterns:
  - `signal.*fault` (0.5)
  - `signal.*error` (0.4)
  - `false.*alarm` (0.4)
  - `wrong.*reading` (0.4)
  - `display.*error` (0.3)
  - `indication.*wrong` (0.4)
  - `sensor.*reading.*incorrect` (0.5)

**Differentiation**: Specifically designed to differentiate from electrical faults (Code: 5.1)

## Test Results

### Expert System Tests: 16/16 passed (100.0%)

**Packing Leak Tests**:
- ✓ "Packing leak detected on pump gland" → 9.1
- ✓ "Stuffing box leaking fluid" → 9.1
- ✓ "Gland packing needs replacement due to leak" → 9.1

**Lubrication Failure Tests**:
- ✓ "Lubrication failure causing bearing damage" → 10.1
- ✓ "Oil level low in lubrication system" → 10.1
- ✓ "Dry running detected due to no lubrication" → 10.1
- ✓ "Grease not flowing to bearing points" → 10.1

**Signal Fault Tests**:
- ✓ "Signal fault in pressure indication" → 11.1
- ✓ "False alarm on temperature sensor" → 11.1
- ✓ "Wrong reading on flow meter display" → 11.1
- ✓ "Sensor reading incorrect values" → 11.1
- ✓ "Display error on control panel" → 11.1

**Differentiation Tests**:
- ✓ "Electrical fault in motor wiring" → 5.1 (not confused with signal fault)
- ✓ "Short circuit in electrical panel" → 5.1 (not confused with signal fault)
- ✓ "Electrical fault causing signal error" → 5.1 (prioritizes electrical fault)

## Integration with AI Classifier

The new rules are fully integrated with the AI classifier system:

1. **Expert System**: Primary rule-based classification
2. **Contextual Patterns**: Enhanced equipment-specific patterns
3. **SpaCy NLP**: Improved entity recognition for new failure types
4. **Sentence Embeddings**: Semantic similarity matching
5. **Hybrid Classification**: Combines all methods for optimal results

## Enhanced Contextual Patterns

Updated equipment contexts to include new failure types:

**Pump Equipment**:
- Added: `packing leak`, `lubrication failure`
- Patterns: `packing.*leak`, `gland.*leak`, `lubrication.*fail`, `oil.*level.*low`

## Files Modified

1. **ai_failure_classifier.py**:
   - Added 3 new expert rules in `_initialize_rules()`
   - Updated `_initialize_failure_patterns()` with new patterns
   - Enhanced contextual patterns for equipment-specific failures

2. **test_new_expert_rules.py**:
   - Comprehensive test suite for new rules
   - Differentiation testing between similar failure types
   - AI classifier integration testing

## Usage

The new expert rules are automatically available when using the AI classifier:

```python
from ai_failure_classifier import AIClassifier

# Initialize classifier
classifier = AIClassifier(confidence_threshold=0.3)

# Load failure dictionary
classifier.load_failure_dictionary('failure_dictionary.xlsx')

# Classify with new rules
result = classifier.classify_hybrid("Packing leak on pump gland")
print(f"Code: {result.code}, Description: {result.description}")
```

## Benefits

1. **Improved Accuracy**: Better differentiation between similar failure types
2. **Enhanced Coverage**: Covers more specific failure scenarios
3. **Equipment-Specific**: Contextual patterns for different equipment types
4. **Maintainable**: Easy to add/modify rules as needed
5. **Integrated**: Works seamlessly with existing AI classification methods

## Future Enhancements

1. **Custom Rules**: Allow users to add custom expert rules
2. **Rule Weights**: Dynamic adjustment of rule weights based on historical data
3. **Equipment-Specific Rules**: More granular rules for specific equipment types
4. **Temporal Patterns**: Time-based rule adjustments for seasonal failures 