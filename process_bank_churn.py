from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def split_train_val(
    raw_df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Розбиває датафрейм на train / val зі стратифікацією за таргетом.

    Args:
        raw_df: Сирі дані, що містять target_col.
        target_col: Назва цільової колонки для стратифікації.
        test_size: Частка валідаційної вибірки.
        random_state: Фіксація випадковості для відтворюваності.

    Returns:
        Кортеж (train_df, val_df).
    """
    train_df, val_df = train_test_split(
        raw_df,
        test_size=test_size,
        stratify=raw_df[target_col],
        random_state=random_state,
    )
    return train_df, val_df


def split_inputs_targets(
    df: pd.DataFrame,
    input_cols: List[str],
    target_col: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Відокремлює ознаки (X) від таргета (y).

    Args:
        df: Датафрейм (train або val).
        input_cols: Назви колонок-ознак.
        target_col: Назва цільової колонки.

    Returns:
        Кортеж (inputs, targets).
    """
    inputs = df[input_cols].copy()
    targets = df[target_col].copy()
    return inputs, targets


def identify_column_types(inputs: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Визначає числові та категоріальні колонки за їх типом.

    Args:
        inputs: Датафрейм з ознаками.

    Returns:
        Кортеж (numeric_cols, categorical_cols).
    """
    numeric_cols = inputs.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = inputs.select_dtypes(include='object').columns.tolist()
    return numeric_cols, categorical_cols


def scale_numeric_features(
    train_inputs: pd.DataFrame,
    val_inputs: pd.DataFrame,
    numeric_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """Навчає MinMaxScaler на train і масштабує числові ознаки train та val.

    Scaler навчається ТІЛЬКИ на train, щоб уникнути витоку даних.

    Args:
        train_inputs: Тренувальні ознаки.
        val_inputs: Валідаційні ознаки.
        numeric_cols: Назви числових колонок.

    Returns:
        Кортеж (train_inputs, val_inputs, scaler) зі масштабованими колонками.
    """
    scaler = MinMaxScaler()
    scaler.fit(train_inputs[numeric_cols])
    train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
    val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
    return train_inputs, val_inputs, scaler


def encode_categorical_features(
    train_inputs: pd.DataFrame,
    val_inputs: pd.DataFrame,
    categorical_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:
    """Навчає OneHotEncoder на train і кодує категоріальні ознаки train та val.

    Encoder навчається ТІЛЬКИ на train. Вихідні категоріальні колонки
    прибираються, замість них додаються one-hot колонки.

    Args:
        train_inputs: Тренувальні ознаки.
        val_inputs: Валідаційні ознаки.
        categorical_cols: Назви категоріальних колонок.

    Returns:
        Кортеж (train_inputs, val_inputs, encoder).
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_inputs[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
    val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
    train_inputs = train_inputs.drop(columns=categorical_cols)
    val_inputs = val_inputs.drop(columns=categorical_cols)
    return train_inputs, val_inputs, encoder


def preprocess_data(
    raw_df: pd.DataFrame,
    scaler_numeric: bool = True,
) -> Dict[str, Union[pd.DataFrame, pd.Series, List[str], MinMaxScaler, OneHotEncoder, None]]:
    """Повна попередня обробка сирих даних змагання.

    Кроки: розбиття на train/val -> (опційно) масштабування числових ознак
    -> one-hot encoding категоріальних ознак. Scaler та encoder навчаються
    тільки на train.

    Args:
        raw_df: Сирі дані з train.csv (мають містити колонку 'Exited').
        scaler_numeric: Якщо True — числові ознаки масштабуються. Для дерева
            можна ставити False (масштабування не впливає на модель).

    Returns:
        Словник з ключами:
            'train_X', 'train_y', 'val_X', 'val_y' — оброблені дані;
            'input_cols' — вихідні (до кодування) назви колонок-ознак;
            'scaler' — навчений MinMaxScaler або None;
            'encoder' — навчений OneHotEncoder.
    """
    target_col = 'Exited'
    input_cols = [
        'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
        'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    ]

    train_df, val_df = split_train_val(raw_df, target_col)

    train_inputs, train_targets = split_inputs_targets(train_df, input_cols, target_col)
    val_inputs, val_targets = split_inputs_targets(val_df, input_cols, target_col)

    numeric_cols, categorical_cols = identify_column_types(train_inputs)

    scaler: Optional[MinMaxScaler] = None
    if scaler_numeric:
        train_inputs, val_inputs, scaler = scale_numeric_features(
            train_inputs, val_inputs, numeric_cols
        )

    train_inputs, val_inputs, encoder = encode_categorical_features(
        train_inputs, val_inputs, categorical_cols
    )

    result = {
        'train_X': train_inputs,
        'train_y': train_targets,
        'val_X': val_inputs,
        'val_y': val_targets,
        'input_cols': input_cols,
        'scaler': scaler,
        'encoder': encoder,
    }
    return result


def preprocess_new_data(
    new_df: pd.DataFrame,
    input_cols: List[str],
    encoder: OneHotEncoder,
    scaler: Optional[MinMaxScaler] = None,
) -> pd.DataFrame:
    """Обробляє нові дані (напр. test.csv) вже навченими scaler та encoder.

    Числові й категоріальні колонки визначаються з навченого encoder, тож
    логіка збігається з тренуванням. Трансформери лише .transform (без .fit),
    тому витоку даних немає.

    Args:
        new_df: Нові сирі дані (мають містити всі колонки з input_cols).
        input_cols: Вихідні назви колонок-ознак (те, що повернув preprocess_data).
        encoder: Навчений OneHotEncoder.
        scaler: Навчений MinMaxScaler або None (якщо масштабування не робили).

    Returns:
        DataFrame з обробленими ознаками, готовий для передбачення моделлю.
        Порядок колонок збігається з 'train_X' із preprocess_data.
    """
    inputs = new_df[input_cols].copy()

    categorical_cols = list(encoder.feature_names_in_)
    numeric_cols = [c for c in input_cols if c not in categorical_cols]

    if scaler is not None:
        inputs[numeric_cols] = scaler.transform(inputs[numeric_cols])

    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    inputs[encoded_cols] = encoder.transform(inputs[categorical_cols])
    inputs = inputs.drop(columns=categorical_cols)

    return inputs
