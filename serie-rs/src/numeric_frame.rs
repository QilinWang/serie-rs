//! Small numeric table boundary for Serie's optional staged pipeline.
//!
//! The scientific runtime only needs named numeric columns, stable row order,
//! row slicing, and explicit column removal. Keeping that contract here avoids
//! linking a general-purpose query engine into Serie or Fern.

use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumericFrameError(String);

impl NumericFrameError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for NumericFrameError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for NumericFrameError {}

pub trait IntoNumericValues {
    fn into_numeric_values(self) -> Vec<Option<f64>>;
}

impl IntoNumericValues for Vec<Option<f64>> {
    fn into_numeric_values(self) -> Vec<Option<f64>> {
        self
    }
}

impl IntoNumericValues for Vec<f64> {
    fn into_numeric_values(self) -> Vec<Option<f64>> {
        self.into_iter().map(Some).collect()
    }
}

impl IntoNumericValues for &[f64] {
    fn into_numeric_values(self) -> Vec<Option<f64>> {
        self.iter().copied().map(Some).collect()
    }
}

impl<const N: usize> IntoNumericValues for &[f64; N] {
    fn into_numeric_values(self) -> Vec<Option<f64>> {
        self.iter().copied().map(Some).collect()
    }
}

impl IntoNumericValues for &Vec<f64> {
    fn into_numeric_values(self) -> Vec<Option<f64>> {
        self.iter().copied().map(Some).collect()
    }
}

impl IntoNumericValues for &[Option<f64>] {
    fn into_numeric_values(self) -> Vec<Option<f64>> {
        self.to_vec()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NumericColumn {
    name: String,
    values: Vec<Option<f64>>,
}

impl NumericColumn {
    pub fn new(name: String, values: impl IntoNumericValues) -> Self {
        Self {
            name,
            values: values.into_numeric_values(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn values(&self) -> &[Option<f64>] {
        &self.values
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn get(&self, index: usize) -> Option<f64> {
        self.values.get(index).copied().flatten()
    }

    pub fn numeric(&self) -> Result<&Self, NumericFrameError> {
        Ok(self)
    }

    pub fn mean(&self) -> Option<f64> {
        let mut count = 0usize;
        let mut sum = 0.0;
        for value in self.values.iter().flatten() {
            count += 1;
            sum += value;
        }
        (count > 0).then(|| sum / count as f64)
    }

    pub fn sample_std(&self) -> Option<f64> {
        let mean = self.mean()?;
        let values: Vec<f64> = self.values.iter().flatten().copied().collect();
        if values.len() < 2 {
            return None;
        }
        let squared_error: f64 = values.iter().map(|value| (value - mean).powi(2)).sum();
        Some((squared_error / (values.len() - 1) as f64).sqrt())
    }

    pub fn min(&self) -> Option<f64> {
        self.values.iter().flatten().copied().reduce(f64::min)
    }

    pub fn max(&self) -> Option<f64> {
        self.values.iter().flatten().copied().reduce(f64::max)
    }

    pub fn quantile_linear(&self, quantile: f64) -> Option<f64> {
        let mut values: Vec<f64> = self.values.iter().flatten().copied().collect();
        if values.is_empty() {
            return None;
        }
        values.sort_by(f64::total_cmp);
        let position = quantile.clamp(0.0, 1.0) * (values.len() - 1) as f64;
        let lower = position.floor() as usize;
        let upper = position.ceil() as usize;
        if lower == upper {
            Some(values[lower])
        } else {
            let weight = position - lower as f64;
            Some(values[lower] + weight * (values[upper] - values[lower]))
        }
    }

    pub fn median(&self) -> Option<f64> {
        self.quantile_linear(0.5)
    }
}

impl<'a> IntoIterator for &'a NumericColumn {
    type Item = Option<f64>;
    type IntoIter = std::iter::Copied<std::slice::Iter<'a, Option<f64>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.iter().copied()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NumericFrame {
    height: usize,
    columns: Vec<NumericColumn>,
}

impl NumericFrame {
    pub fn new(height: usize, columns: Vec<NumericColumn>) -> Result<Self, NumericFrameError> {
        if columns.iter().any(|column| column.len() != height) {
            return Err(NumericFrameError::new(format!(
                "numeric frame column length does not match declared height {height}"
            )));
        }
        let mut names = std::collections::HashSet::with_capacity(columns.len());
        if columns.iter().any(|column| !names.insert(column.name().to_string())) {
            return Err(NumericFrameError::new("numeric frame column names must be unique"));
        }
        Ok(Self { height, columns })
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn width(&self) -> usize {
        self.columns.len()
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.height, self.width())
    }

    pub fn column(&self, name: &str) -> Result<&NumericColumn, NumericFrameError> {
        self.columns
            .iter()
            .find(|column| column.name() == name)
            .ok_or_else(|| NumericFrameError::new(format!("numeric frame has no column {name:?}")))
    }

    pub fn column_names(&self) -> Vec<String> {
        self.columns.iter().map(|column| column.name().to_string()).collect()
    }

    pub fn replace(&mut self, name: &str, column: NumericColumn) -> Result<(), NumericFrameError> {
        if column.len() != self.height {
            return Err(NumericFrameError::new(format!(
                "replacement column {:?} has {} rows; expected {}",
                column.name(),
                column.len(),
                self.height
            )));
        }
        let index = self
            .columns
            .iter()
            .position(|existing| existing.name() == name)
            .ok_or_else(|| NumericFrameError::new(format!("numeric frame has no column {name:?}")))?;
        self.columns[index] = column;
        Ok(())
    }

    pub fn drop_columns(&self, names: &[String]) -> Self {
        let columns = self
            .columns
            .iter()
            .filter(|column| !names.iter().any(|name| name == column.name()))
            .cloned()
            .collect();
        Self {
            height: self.height,
            columns,
        }
    }

    pub fn drop_column(&mut self, name: &str) -> Result<NumericColumn, NumericFrameError> {
        let index = self
            .columns
            .iter()
            .position(|column| column.name() == name)
            .ok_or_else(|| NumericFrameError::new(format!("numeric frame has no column {name:?}")))?;
        Ok(self.columns.remove(index))
    }

    pub fn drop_null_rows(&self) -> Result<Self, NumericFrameError> {
        let kept_rows: Vec<usize> = (0..self.height)
            .filter(|row| self.columns.iter().all(|column| column.values[*row].is_some()))
            .collect();
        let columns = self
            .columns
            .iter()
            .map(|column| {
                NumericColumn::new(
                    column.name.clone(),
                    kept_rows.iter().map(|row| column.values[*row]).collect::<Vec<_>>(),
                )
            })
            .collect();
        Self::new(kept_rows.len(), columns)
    }

    pub fn slice(&self, start: usize, length: usize) -> Result<Self, NumericFrameError> {
        let end = start
            .checked_add(length)
            .filter(|end| *end <= self.height)
            .ok_or_else(|| NumericFrameError::new("numeric frame slice is out of bounds"))?;
        let columns = self
            .columns
            .iter()
            .map(|column| NumericColumn::new(column.name.clone(), column.values[start..end].to_vec()))
            .collect();
        Self::new(length, columns)
    }

    pub fn row_major_f64(&self) -> Vec<f64> {
        let mut values = Vec::with_capacity(self.height * self.width());
        for row in 0..self.height {
            for column in &self.columns {
                values.push(column.values[row].unwrap_or(0.0));
            }
        }
        values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_preserves_named_columns_and_row_major_order() {
        let frame = NumericFrame::new(
            2,
            vec![
                NumericColumn::new("a".to_string(), &[1.0, 2.0][..]),
                NumericColumn::new("b".to_string(), &[10.0, 20.0][..]),
            ],
        )
        .unwrap();

        assert_eq!(frame.column_names(), ["a", "b"]);
        assert_eq!(frame.row_major_f64(), [1.0, 10.0, 2.0, 20.0]);
    }

    #[test]
    fn linear_quantile_matches_the_partition_contract() {
        let column = NumericColumn::new("x".to_string(), &[0.0, 10.0, 20.0, 30.0][..]);
        assert_eq!(column.quantile_linear(0.25), Some(7.5));
        assert_eq!(column.median(), Some(15.0));
    }
}
