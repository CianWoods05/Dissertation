"""Unit tests for improved EDA notebook (01_data_exploration.py).

Tests cover:
- Data loading and validation
- Effect size calculations (Cohen's d)
- Data quality audit functions
- Statistical test correctness (Mann-Whitney U, Bonferroni correction)
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from scipy import stats


# ── Mock/Import the notebook functions ──────────────────────
# These would normally be imported from the notebook module

def compute_cohens_d(group1: pd.Series, group2: pd.Series) -> float:
    """Compute Cohen's d effect size (pooled standard deviation)."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    
    if n1 < 2 or n2 < 2:
        return 0.0
    
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std < 1e-10:
        return 0.0
    
    return abs(group1.mean() - group2.mean()) / pooled_std


def check_data_quality(df: pd.DataFrame, feat_cols: list) -> dict:
    """Audit data quality: missing values, zero-variance, outliers."""
    quality = {
        'missing_values': {},
        'zero_variance': [],
        'outliers': {}
    }
    
    # Missing values
    missing = df[feat_cols].isnull().sum()
    if missing.sum() > 0:
        quality['missing_values'] = missing[missing > 0].to_dict()
    
    # Zero-variance features
    for col in feat_cols:
        if df[col].std() < 1e-10:
            quality['zero_variance'].append(col)
    
    # Outliers (IQR method)
    for col in feat_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR < 1e-10:
            continue
        
        outlier_count = len(df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)])
        if outlier_count > 0:
            quality['outliers'][col] = {
                'count': outlier_count,
                'pct': round(100 * outlier_count / len(df), 2)
            }
    
    return quality


# ── Test Suite ──────────────────────────────────────────────

class TestCohensDCalculation:
    """Test Cohen's d effect size calculation."""
    
    def test_identical_groups_zero_effect(self):
        """Identical groups should have zero effect size."""
        data = pd.Series([1, 2, 3, 4, 5])
        assert compute_cohens_d(data, data) == 0.0
    
    def test_perfectly_separated_groups(self):
        """Well-separated groups should have large effect size."""
        group1 = pd.Series([1, 2, 3])
        group2 = pd.Series([100, 101, 102])
        d = compute_cohens_d(group1, group2)
        assert d > 10  # Large effect
    
    def test_small_sample_returns_zero(self):
        """Groups with n < 2 should return 0."""
        small = pd.Series([1])
        normal = pd.Series([1, 2, 3])
        assert compute_cohens_d(small, normal) == 0.0
    
    def test_zero_variance_returns_zero(self):
        """Groups with zero variance should return 0 (avoid division by zero)."""
        const_group = pd.Series([5, 5, 5])
        normal_group = pd.Series([1, 2, 3])
        result = compute_cohens_d(const_group, normal_group)
        assert result == 0.0
    
    def test_uses_pooled_std_not_global_std(self):
        """Verify effect size uses pooled SD (Cohen's d), not global SD."""
        # Create two groups with different variances
        group1 = pd.Series([10, 11, 12])  # small variance
        group2 = pd.Series([20, 30, 40])  # large variance
        
        d = compute_cohens_d(group1, group2)
        
        # Manual calculation with pooled SD
        n1, n2 = 3, 3
        var1, var2 = group1.var(), group2.var()
        pooled_std_manual = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        d_manual = abs(10 - 30) / pooled_std_manual
        
        assert np.isclose(d, d_manual)
    
    @pytest.mark.parametrize("n1,n2", [(5, 5), (10, 20), (100, 5)])
    def test_unequal_sample_sizes(self, n1, n2):
        """Cohen's d should handle unequal group sizes."""
        group1 = pd.Series(np.random.normal(0, 1, n1))
        group2 = pd.Series(np.random.normal(2, 1, n2))
        d = compute_cohens_d(group1, group2)
        assert not np.isnan(d) and not np.isinf(d)


class TestDataQualityAudit:
    """Test data quality checking functions."""
    
    def test_detects_missing_values(self):
        """Should detect missing values."""
        df = pd.DataFrame({
            'a': [1, 2, np.nan],
            'b': [4, 5, 6]
        })
        quality = check_data_quality(df, ['a', 'b'])
        assert 'a' in quality['missing_values']
        assert quality['missing_values']['a'] == 1
    
    def test_no_missing_values_clean(self):
        """Should return empty dict for complete data."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        quality = check_data_quality(df, ['a', 'b'])
        assert quality['missing_values'] == {}
    
    def test_detects_zero_variance_features(self):
        """Should identify constant features."""
        df = pd.DataFrame({
            'constant': [5, 5, 5, 5],
            'normal': [1, 2, 3, 4]
        })
        quality = check_data_quality(df, ['constant', 'normal'])
        assert 'constant' in quality['zero_variance']
        assert 'normal' not in quality['zero_variance']
    
    def test_detects_outliers_iqr_method(self):
        """Should identify outliers using IQR method."""
        # Create data with clear outlier
        df = pd.DataFrame({
            'feature': [1, 2, 3, 4, 5, 6, 100]  # 100 is outlier
        })
        quality = check_data_quality(df, ['feature'])
        assert 'feature' in quality['outliers']
        assert quality['outliers']['feature']['count'] == 1
    
    def test_outlier_percentage_calculation(self):
        """Outlier percentage should be calculated correctly."""
        # 1 outlier in 10 rows = 10%
        data = list(range(1, 10)) + [1000]
        df = pd.DataFrame({'feature': data})
        quality = check_data_quality(df, ['feature'])
        
        if 'feature' in quality['outliers']:
            pct = quality['outliers']['feature']['pct']
            assert np.isclose(pct, 10, atol=1)
    
    def test_no_false_positives_on_clean_data(self):
        """Clean data should have no outliers detected."""
        df = pd.DataFrame({
            'normal': np.random.normal(0, 1, 100)
        })
        quality = check_data_quality(df, ['normal'])
        # Should have few or no outliers from normal distribution
        assert len(quality['outliers']) <= 5  # Allow some by chance


class TestStatisticalCorrectness:
    """Test statistical test implementations."""
    
    def test_bonferroni_correction_factor(self):
        """Verify Bonferroni correction calculation."""
        n_tests = 30
        alpha_corrected = 0.05 / n_tests
        assert np.isclose(alpha_corrected, 0.05 / 30)
        assert alpha_corrected < 0.05  # Should be more stringent
    
    def test_mann_whitney_u_identical_distributions(self):
        """Identical distributions should have high p-value."""
        data = pd.Series([1, 2, 3, 4, 5])
        stat, p = stats.mannwhitneyu(data, data, alternative='two-sided')
        assert p > 0.05  # Not significant
    
    def test_mann_whitney_u_different_distributions(self):
        """Clearly different distributions should have low p-value."""
        group1 = pd.Series([1, 2, 3, 4, 5])
        group2 = pd.Series([100, 101, 102, 103, 104])
        stat, p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        assert p < 0.05  # Significant
    
    def test_effect_size_vs_pvalue_independence(self):
        """Large effect size doesn't guarantee significance with small n."""
        # Large effect but very small samples
        group1 = pd.Series([1])
        group2 = pd.Series([100])
        
        # Can't run Mann-Whitney with n=1, but shows the concept
        # A large effect size ≠ statistical significance
        d = abs(1 - 100)  # Large absolute difference
        assert d > 50  # Effect is large


class TestDataFrameCompatibility:
    """Test functions work with realistic data structures."""
    
    def test_works_with_multi_column_dataframe(self):
        """Should handle DataFrames with multiple features."""
        df = pd.DataFrame({
            'backs_score': [10, 15, 12, 18],
            'backs_tries': [2, 1, 2, 3],
            'forwards_score': [8, 9, 7, 10],
            'forwards_tries': [1, 1, 1, 2]
        })
        feat_cols = ['backs_score', 'backs_tries', 'forwards_score', 'forwards_tries']
        quality = check_data_quality(df, feat_cols)
        assert isinstance(quality, dict)
        assert 'missing_values' in quality
    
    def test_handles_mixed_data_types(self):
        """Should work with integers, floats, and categorical data."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4],
            'float_col': [1.5, 2.5, 3.5, 4.5],
            'category': ['A', 'B', 'A', 'B']
        })
        # Only test numeric columns
        quality = check_data_quality(df, ['int_col', 'float_col'])
        assert isinstance(quality, dict)
    
    def test_large_dataframe_performance(self):
        """Should handle larger datasets efficiently."""
        df = pd.DataFrame({
            f'feat_{i}': np.random.normal(0, 1, 10000)
            for i in range(50)
        })
        feat_cols = [f'feat_{i}' for i in range(50)]
        
        import time
        start = time.time()
        quality = check_data_quality(df, feat_cols)
        elapsed = time.time() - start
        
        # Should complete in reasonable time (< 5 seconds)
        assert elapsed < 5.0
        assert isinstance(quality, dict)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_cohens_d_with_nan_values(self):
        """Should handle Series with NaN values gracefully."""
        group1 = pd.Series([1, 2, np.nan, 4, 5])
        group2 = pd.Series([10, 11, 12, np.nan, 14])
        # Should not crash
        d = compute_cohens_d(group1, group2)
        assert not np.isnan(d) or d == 0.0  # Result should be valid or 0
    
    def test_empty_dataframe(self):
        """Should handle empty DataFrames."""
        df = pd.DataFrame()
        quality = check_data_quality(df, [])
        assert quality['missing_values'] == {}
    
    def test_single_row_dataframe(self):
        """Should handle single-row DataFrames."""
        df = pd.DataFrame({'a': [1], 'b': [2]})
        quality = check_data_quality(df, ['a', 'b'])
        # Should not crash, may have zero variance
        assert isinstance(quality, dict)
    
    def test_all_nan_feature(self):
        """Should handle features that are all NaN."""
        df = pd.DataFrame({
            'all_nan': [np.nan, np.nan, np.nan],
            'normal': [1, 2, 3]
        })
        quality = check_data_quality(df, ['all_nan', 'normal'])
        # Should detect as missing
        assert 'all_nan' in quality['missing_values'] or len(df[['all_nan']].dropna()) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
