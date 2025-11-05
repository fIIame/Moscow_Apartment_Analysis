from typing import Optional

import pandas as pd
from scipy.stats import (
    spearmanr, pearsonr,
    mannwhitneyu, kruskal,
    shapiro
)


def _is_normal(column: pd.Series, alpha=0.05) -> bool:
    stat, p = shapiro(column)
    if p >= alpha:
        return True
    else:
        return False


def get_quant_p_values(df: pd.DataFrame, target: str, factors: Optional[pd.Index] = None) -> pd.DataFrame:
    """
    Рассчитывает коэффициенты корреляции и p-values между целевой переменной и количественными признаками.

    Параметры
    ----------
    df : pd.DataFrame
        Датасет с данными.
    target : str
        Название целевого признака, относительно которого проверяем корреляцию.
    factors : Optional[pd.Index], default=None
        Список признаков для проверки. Если None, берутся все числовые колонки, кроме target.

    Возвращает
    ----------
    pd.DataFrame
        Таблица с колонками: 'column', 'p_value', 'correlation', 'method', 'conclusion'.
    """
    # Если не указаны признаки, выбираем все числовые, кроме целевого
    if factors is None:
        factors = df.select_dtypes(exclude='object').columns.drop(target)

    rows = []
    for factor in factors:
        # Определяем, какая корреляция: Пирсона или Спирмена
        if _is_normal(df[factor]) and _is_normal(df[target]):
            r, p = pearsonr(df[factor], df[target])
            method = 'Pearson'
        else:
            r, p = spearmanr(df[factor], df[target])
            method = 'Spearman'

        # Делаем вывод по значимости
        if p <= 0.05:
            conclusion = "Есть статистически значимая разница."
        else:
            conclusion = "Нет статистически значимой разницы."

        rows.append({
            "column": factor,
            "p_value": round(p, 4),
            "correlation": round(r, 4),
            "method": method,
            "conclusion": conclusion
        })

    return pd.DataFrame(rows)


def test_mannwhitney(df: pd.DataFrame, target: str, factor: str, alternative: str) -> None:
    """
    Выполняет тест Манна–Уитни для сравнения двух групп по количественному признаку.

    Параметры:
    ----------
    df : pd.DataFrame
        Датасет с данными.
    target : str
        Название количественного признака.
    factor : str
        Название категориального признака с двумя группами.
    alternative : str
        Альтернатива: 'two-sided', 'less', 'greater'.
    """
    groups = df.groupby(factor)[target].apply(list)
    stat, p = mannwhitneyu(*groups, alternative=alternative)
    if p >= 0.05:
        msg = f"{factor}: p-value={p:.4f} → Нет статистически значимой разницы."
    else:
        msg = f"{factor}: p-value={p:.4f} → Есть статистически значимая разница."

    print(msg)


def test_kruskal(df: pd.DataFrame, target: str, factor: str) -> None:
    """
    Выполняет тест Краскала–Уоллиса для сравнения нескольких групп по количественному признаку.

    Параметры:
    ----------
    df : pd.DataFrame
        Датасет с данными.
    target : str
        Название количественного признака.
    factor : str
        Название категориального признака.
    """
    groups = df.groupby(factor)[target].apply(list)
    stat, p = kruskal(*groups)
    if p >= 0.05:
        msg = f"{factor}: p-value={p:.4f} → Нет статистически значимой разницы."
    else:
        msg = f"{factor}: p-value={p:.4f} → Есть статистически значимая разница."

    print(msg)