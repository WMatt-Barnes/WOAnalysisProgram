# Enhanced AI Failure Mode Classifier

## Overview

The Enhanced AI Failure Mode Classifier is a sophisticated system that combines multiple classification methods to intelligently assign failure codes to work order descriptions. This system removes dependency on external APIs (like OpenAI) and focuses on local, high-performance classification methods.

## Key Features

### 🧠 **Expert System (Rule-Based Classification)**
- **Purpose**: Provides deterministic, rule-based classification using industry knowledge
- **Performance**: Very fast (<1ms per classification)
- **Memory**: Minimal (<1MB)
- **Accuracy**: High for well-defined patterns
- **Use Cases**: Standard failure modes with clear symptoms

**Example Rules:**
- Bearing failures: Keywords + patterns like "bearing.*noise", "bearing.*vibration"
- Seal leaks: "seal.*leak", "packing.*leak", "mechanical.*seal"
- Motor overheating: "motor.*overheat", "temperature.*high", "thermal.*trip"

### 🔍 **Contextual Pattern Recognition**
- **Purpose**: Equipment-specific failure pattern recognition
- **Performance**: Fast (1-5ms per classification)
- **Memory**: Low (<5MB)
- **Accuracy**: High for equipment-specific contexts
- **Use Cases**: Equipment with known failure patterns

**Supported Equipment Types:**
- **Pumps**: Cavitation, seal leaks, bearing failures, impeller damage
- **Motors**: Overheating, electrical faults, insulation failures
- **Valves**: Stuck/seized, leaks, actuator failures
- **Compressors**: Overheating, vibration, bearing failures
- **Fans**: Blade damage, bearing failures, imbalance

### 📈 **Temporal Analysis**
- **Purpose**: Historical pattern analysis for recurring failures
- **Performance**: Medium (5-10ms per classification)
- **Memory**: Variable (depends on historical data size)
- **Accuracy**: High for recurring patterns
- **Use Cases**: Equipment with historical failure data

**Features:**
- Analyzes failure patterns over time
- Identifies recurring failure types per equipment
- Uses similarity matching for historical descriptions
- Provides confidence based on pattern strength

### 🤖 **Sentence Embeddings (Semantic Similarity)**
- **Purpose**: Semantic understanding of failure descriptions
- **Performance**: Medium (10-50ms per classification)
- **Memory**: ~500MB (model size)
- **Accuracy**: High for semantic understanding
- **Use Cases**: Complex or ambiguous descriptions

### 📝 **SpaCy NLP Analysis**
- **Purpose**: Advanced linguistic analysis and entity recognition
- **Performance**: Medium (20-100ms per classification)
- **Memory**: ~50MB (model size)
- **Accuracy**: High for linguistic features
- **Use Cases**: Complex descriptions with technical terms

### 📚 **Dictionary Fallback**
- **Purpose**: Traditional keyword and fuzzy matching
- **Performance**: Fast (1-10ms per classification)
- **Memory**: Minimal (<1MB)
- **Accuracy**: Medium for exact matches
- **Use Cases**: Fallback when AI methods fail

## Performance Characteristics

| Method | Speed | Memory | Accuracy | Use Case |
|--------|-------|--------|----------|----------|
| Expert System | <1ms | <1MB | High | Standard patterns |
| Contextual Patterns | 1-5ms | <5MB | High | Equipment-specific |
| Temporal Analysis | 5-10ms | Variable | High | Historical patterns |
| Sentence Embeddings | 10-50ms | ~500MB | High | Semantic understanding |
| SpaCy NLP | 20-100ms | ~50MB | High | Linguistic analysis |
| Dictionary Fallback | 1-10ms | <1MB | Medium | Exact matches |

## Installation

### Prerequisites
```bash
pip install pandas numpy sentence-transformers spacy rapidfuzz
python -m spacy download en_core_web_sm
```

### Optional Dependencies
```bash
pip install scikit-learn matplotlib seaborn
```

## Usage

### Basic Usage

```python
from ai_failure_classifier import AIClassifier

# Initialize the classifier
ai_classifier = AIClassifier(
    confidence_threshold=0.7,
    cache_file="ai_classification_cache.json"
)

# Load failure dictionary
ai_classifier.load_failure_dictionary("failure_mode_dictionary.xlsx")

# Classify a single description
result = ai_classifier.classify_hybrid(
    "Bearing making loud noise and excessive vibration",
    dictionary_fallback_func
)

print(f"Method: {result.method}")
print(f"Code: {result.code}")
print(f"Description: {result.description}")
print(f"Confidence: {result.confidence}")
```

### Batch Processing

```python
# Analyze historical patterns first
ai_classifier.analyze_historical_patterns(work_orders_df)

# Batch classify multiple descriptions
descriptions = [
    "Bearing noise and vibration",
    "Seal leaking around shaft",
    "Motor overheating"
]

results = ai_classifier.batch_classify(descriptions, dictionary_fallback_func)

for result in results:
    print(f"{result.method}: {result.code} (confidence: {result.confidence:.3f})")
```

### Individual Method Usage

```python
from ai_failure_classifier import (
    ExpertSystemClassifier, 
    ContextualPatternClassifier, 
    TimeSeriesPatternClassifier
)

# Expert System
expert = ExpertSystemClassifier()
code, desc, conf = expert.classify("Bearing making noise")

# Contextual Patterns
contextual = ContextualPatternClassifier()
code, desc, conf = contextual.classify_with_context("Pump cavitation noise")

# Temporal Analysis
temporal = TimeSeriesPatternClassifier()
temporal.analyze_temporal_patterns(work_orders_df)
code, desc, conf = temporal.classify_with_temporal_context(
    "Bearing noise again", "PUMP-001", datetime.now()
)
```

## Configuration

### Expert System Rules

The expert system uses weighted rules with conditions:

```python
{
    'name': 'bearing_failure',
    'conditions': [
        {'type': 'keyword', 'value': 'bearing', 'weight': 0.3},
        {'type': 'keyword', 'value': 'noise', 'weight': 0.2},
        {'type': 'pattern', 'value': r'bearing.*fail', 'weight': 0.4}
    ],
    'failure_code': '1.1',
    'description': 'Bearing Failure'
}
```

### Contextual Patterns

Equipment-specific patterns are defined per equipment type:

```python
'pump': {
    'common_failures': ['cavitation', 'seal leak', 'bearing failure'],
    'context_words': ['suction', 'discharge', 'flow', 'pressure'],
    'failure_patterns': {
        'cavitation': [r'cavitation', r'noise.*suction'],
        'seal_leak': [r'seal.*leak', r'packing.*leak']
    }
}
```

## Integration with Main Application

The enhanced AI classifier is fully integrated into the Work Order Analysis application:

1. **AI Settings Tab**: Configure confidence thresholds and enable/disable methods
2. **Processing**: Automatically uses the best available method for each description
3. **Statistics**: Track performance and method usage
4. **Caching**: Persistent cache for improved performance
5. **Export**: Export training data for analysis

## Performance Optimization

### Caching Strategy
- **Embeddings**: Cached to avoid recomputation
- **Expert System**: Results cached for repeated patterns
- **Temporal Analysis**: Historical patterns cached after analysis

### Batch Processing
- **Parallel Processing**: Multiple descriptions processed efficiently
- **Memory Management**: Optimized for large datasets
- **Progress Tracking**: Real-time progress updates

### Memory Usage
- **Lazy Loading**: Models loaded only when needed
- **Garbage Collection**: Automatic cleanup of unused data
- **Memory Monitoring**: Track memory usage during processing

## Testing

Run the test suite to verify functionality:

```bash
python test_enhanced_ai.py
```

The test suite covers:
- Expert System classification
- Contextual Pattern recognition
- Temporal Analysis
- Complete AI Classifier integration
- Performance benchmarks

## Troubleshooting

### Common Issues

1. **SpaCy Model Not Found**
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **Sentence Transformers Not Available**
   ```bash
   pip install sentence-transformers
   ```

3. **Memory Issues with Large Datasets**
   - Reduce batch size
   - Use dictionary fallback only
   - Increase system memory

4. **Low Classification Accuracy**
   - Adjust confidence thresholds
   - Review failure dictionary
   - Check for data quality issues

### Performance Tuning

1. **For Speed**: Use Expert System + Contextual Patterns only
2. **For Accuracy**: Enable all methods with lower confidence thresholds
3. **For Memory**: Use dictionary fallback + Expert System only
4. **For Large Datasets**: Use batch processing with appropriate chunk sizes

## Future Enhancements

### Planned Features
- **Machine Learning Models**: Train custom models on historical data
- **Anomaly Detection**: Identify unusual failure patterns
- **Predictive Analytics**: Predict future failures based on patterns
- **Multi-language Support**: Support for non-English descriptions
- **Real-time Learning**: Continuous improvement from new data

### Performance Improvements
- **GPU Acceleration**: Use GPU for embedding computations
- **Distributed Processing**: Scale across multiple machines
- **Streaming Processing**: Real-time classification of incoming data
- **Model Optimization**: Quantized models for faster inference

## Contributing

1. **Code Style**: Follow PEP 8 guidelines
2. **Testing**: Add tests for new features
3. **Documentation**: Update documentation for changes
4. **Performance**: Benchmark new methods
5. **Compatibility**: Ensure backward compatibility

## License

This enhanced AI classifier is part of the Work Order Analysis system and follows the same licensing terms. 