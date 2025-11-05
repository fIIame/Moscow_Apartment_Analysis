from typing import Optional

import pandas as pd
from scipy.stats import shapiro


def drop_outliers(
        data: pd.DataFrame,
        column: str,
        k: float = 1.5,
        inplace: bool = True
) -> Optional[pd.DataFrame]:
    """
    Удаляет выбросы из указанных колонок с помощью правила IQR.

    Параметры:
    ----------
    data : pd.DataFrame
        Датафрейм с данными.
    columns : str или list[str]
        Одна колонка или список колонок для очистки.
    k : float, default=1.5
        Множитель для определения границ выбросов.
    inplace : bool, default=True
        Если True, изменения вносятся в исходный DataFrame, иначе возвращается новый.

    Возвращает:
    ----------
    pd.DataFrame или None
        Возвращает новый DataFrame, если inplace=False, иначе None.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR

    if inplace:
        data.query(f"{column} >= @lower_bound and {column} <= @upper_bound", inplace=True)
        return None
    else:
        return data.query(f"{column} >= @lower_bound and {column} <= @upper_bound")


def drop_outliers_grouped(
        data: pd.DataFrame,
        column: str,
        group_col: str,
        k: float = 1.5,
        inplace: bool = True
) -> Optional[pd.DataFrame]:
    """
    Удаляет выбросы в числовом столбце отдельно для каждой группы (по правилу IQR).

    Параметры:
    ----------
    data : pd.DataFrame
        Датафрейм с данными.
    column : str
        Числовая колонка, в которой ищем выбросы.
    group_col : str
        Категориальная колонка, по которой группируем данные.
    k : float, default=1.5
        Коэффициент для расчёта границ выбросов.
    inplace : bool, default=True
        Если True, изменения применяются к исходному DataFrame.

    Возвращает:
    ----------
    pd.DataFrame или None
        Новый DataFrame, если inplace=False, иначе None.
    """

    cleaned = []

    for name, group in data.groupby(group_col):
        Q1 = group[column].quantile(0.25)
        Q3 = group[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - k * IQR
        upper = Q3 + k * IQR
        group_clean = group[(group[column] >= lower) & (group[column] <= upper)]
        cleaned.append(group_clean)

    result = pd.concat(cleaned).reset_index(drop=True)

    if inplace:
        data.loc[:] = result
        return None
    else:
        return result



def check_normality(df: pd.DataFrame, columns: Optional[pd.Index] = None, alpha=0.05):
    """
    Проверяет нормальность распределения количественных признаков.

    Параметры:
    ----------
    df : pd.DataFrame
        Датафрейм с данными.
    columns : list[str] или pd.Index, default=None
        Список колонок для проверки. Если None, проверяются все числовые.
    alpha : float, default=0.05
        Уровень значимости для теста.

    Возвращает:
    ----------
    pd.DataFrame
        DataFrame с колонками: column, p_value, distribution
    """
    if columns is None:
        columns = df.select_dtypes(exclude="object").columns

    rows = []
    for col in columns:
        stat, p = shapiro(df[col].dropna())
        result = "Нормальное" if p >= alpha else "Асимметричное"
        rows.append({"column": col, "p_value": round(p, 4), "distribution": result})

    return pd.DataFrame(rows)
