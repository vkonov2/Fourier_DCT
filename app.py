import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.fft import rfft, irfft, dct, idct
import pandas as pd

def load_rhos_data(filename='func.txt'):
    """Загружает данные из файла func.txt"""
    try:
        with open(filename, 'r') as f:
            content = f.read().strip()
        
        data = []
        for x in content.split(','):
            x = x.strip()
            if x:
                try:
                    data.append(float(x))
                except ValueError:
                    print(f"Предупреждение: не удалось преобразовать '{x}' в число, пропускаем")
        
        return np.array(data)
    except FileNotFoundError:
        st.error(f"Файл {filename} не найден!")
        return None

def fourier_reconstruction(f, num_coeffs):
    """Восстановление сигнала с помощью Фурье преобразования"""
    N = len(f)
    f_fft = rfft(f)
    energies = np.abs(f_fft)
    idx_top = np.argsort(energies)[-num_coeffs:][::-1]
    
    f_fft_approx = np.zeros_like(f_fft)
    f_fft_approx[idx_top] = f_fft[idx_top]
    f_approx = irfft(f_fft_approx, n=N)
    
    err = np.linalg.norm(f - f_approx) / np.linalg.norm(f)
    return f_approx, err, f_fft, energies, idx_top

def dct_reconstruction(f, num_coeffs):
    """Восстановление сигнала с помощью DCT преобразования"""
    f_dct = dct(f, type=2, norm='ortho')
    f_dct_approx = np.zeros_like(f_dct)
    f_dct_approx[:num_coeffs] = f_dct[:num_coeffs]
    f_approx = idct(f_dct_approx, type=2, norm='ortho')
    
    err = np.linalg.norm(f - f_approx) / np.linalg.norm(f)
    return f_approx, err, f_dct

def plot_reconstruction_comparison(f, f_approx_fft, f_approx_dct, err_fft, err_dct):
    """Построение графиков сравнения восстановления с помощью Plotly"""
    x = np.arange(len(f))
    
    # Создаем подграфики
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Сравнение восстановления сигнала', 'Абсолютные ошибки восстановления'),
        vertical_spacing=0.1
    )
    
    # График восстановления
    fig.add_trace(
        go.Scatter(x=x, y=f, mode='lines', name='Исходный сигнал', 
                  line=dict(color='blue', width=2), opacity=0.8),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x, y=f_approx_fft, mode='lines', 
                  name=f'Фурье (ошибка: {err_fft:.4f})',
                  line=dict(color='red', width=2, dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x, y=f_approx_dct, mode='lines', 
                  name=f'DCT (ошибка: {err_dct:.4f})',
                  line=dict(color='green', width=2, dash='dash')),
        row=1, col=1
    )
    
    # График ошибок восстановления
    error_fft = np.abs(f - f_approx_fft)
    error_dct = np.abs(f - f_approx_dct)
    
    fig.add_trace(
        go.Scatter(x=x, y=error_fft, mode='lines', 
                  name=f'Ошибка Фурье (средняя: {np.mean(error_fft):.4f})',
                  line=dict(color='red', width=1)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=x, y=error_dct, mode='lines', 
                  name=f'Ошибка DCT (средняя: {np.mean(error_dct):.4f})',
                  line=dict(color='green', width=1)),
        row=2, col=1
    )
    
    # Настройка макета
    fig.update_layout(
        height=1000,
        showlegend=True,
        title_text="Сравнение методов восстановления сигнала",
        title_x=0.5
    )
    
    # Настройка осей
    fig.update_xaxes(title_text="Индекс", row=1, col=1)
    fig.update_yaxes(title_text="Значение", row=1, col=1)
    fig.update_xaxes(title_text="Индекс", row=2, col=1)
    fig.update_yaxes(title_text="Абсолютная ошибка", row=2, col=1)
    
    return fig

def plot_spectrum_comparison(f_fft, energies, idx_top, f_dct, num_coeffs_fft, num_coeffs_dct):
    """Построение графиков спектров с помощью Plotly"""
    freq = np.arange(len(energies))
    dct_coeffs = np.arange(len(f_dct))
    
    # Создаем подграфики
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Спектр Фурье (модули коэффициентов)', 'Спектр DCT (модули коэффициентов)'),
        vertical_spacing=0.1
    )
    
    # Спектр Фурье
    fig.add_trace(
        go.Scatter(x=freq, y=energies, mode='lines', name='Полный спектр',
                  line=dict(color='blue', width=1), opacity=0.7),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=freq[idx_top], y=energies[idx_top], mode='markers', 
                  name=f'Выбранные {num_coeffs_fft} коэффициентов',
                  marker=dict(color='red', size=8, symbol='circle')),
        row=1, col=1
    )
    
    # Спектр DCT
    fig.add_trace(
        go.Scatter(x=dct_coeffs, y=np.abs(f_dct), mode='lines', name='Полный спектр',
                  line=dict(color='green', width=1), opacity=0.7),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=dct_coeffs[:num_coeffs_dct], y=np.abs(f_dct[:num_coeffs_dct]), 
                  mode='markers', name=f'Выбранные {num_coeffs_dct} коэффициентов',
                  marker=dict(color='red', size=8, symbol='circle')),
        row=2, col=1
    )
    
    # Настройка макета
    fig.update_layout(
        height=900,
        showlegend=True,
        title_text="Спектральный анализ",
        title_x=0.5
    )
    
    # Настройка осей с логарифмической шкалой
    fig.update_xaxes(title_text="Частота", row=1, col=1)
    fig.update_yaxes(title_text="Модуль", type="log", row=1, col=1)
    fig.update_xaxes(title_text="Индекс коэффициента", row=2, col=1)
    fig.update_yaxes(title_text="Модуль", type="log", row=2, col=1)
    
    return fig

def plot_coefficients_comparison(f_fft, energies, idx_top, f_dct, num_coeffs_fft, num_coeffs_dct):
    """Дополнительная визуализация сравнения коэффициентов"""
    # Создаем DataFrame для визуализации
    fourier_data = pd.DataFrame({
        'Индекс': idx_top,
        'Модуль': energies[idx_top],
        'Метод': 'Фурье'
    })
    
    dct_data = pd.DataFrame({
        'Индекс': range(num_coeffs_dct),
        'Модуль': np.abs(f_dct[:num_coeffs_dct]),
        'Метод': 'DCT'
    })
    
    combined_data = pd.concat([fourier_data, dct_data], ignore_index=True)
    
    # Создаем график
    fig = px.bar(combined_data, x='Индекс', y='Модуль', color='Метод',
                 title='Сравнение выбранных коэффициентов',
                 barmode='group')
    
    fig.update_layout(
        height=800,
        title_x=0.5,
        xaxis_title="Индекс коэффициента",
        yaxis_title="Модуль",
        yaxis_type="log"
    )
    
    return fig

# Настройка страницы Streamlit
st.set_page_config(page_title="Визуализация восстановления сигнала", layout="wide")
st.title("🎯 Визуализация восстановления сигнала: Фурье vs DCT")

# Загрузка данных
f = load_rhos_data()
if f is None:
    st.stop()

st.success(f"✅ Загружено {len(f)} точек данных")

# Боковая панель с настройками
st.sidebar.header("⚙️ Настройки параметров")

# Слайдеры для количества коэффициентов
max_coeffs_fft = min(50, len(f) // 2)  # Максимум для Фурье
max_coeffs_dct = min(100, len(f))      # Максимум для DCT

num_coeffs_fft = st.sidebar.slider(
    "Количество коэффициентов Фурье", 
    min_value=1, 
    max_value=max_coeffs_fft, 
    value=12,
    help="Количество самых значимых комплексных коэффициентов Фурье"
)

num_coeffs_dct = st.sidebar.slider(
    "Количество коэффициентов DCT", 
    min_value=1, 
    max_value=max_coeffs_dct, 
    value=24,
    help="Количество первых коэффициентов DCT"
)

# Выполнение восстановления
f_approx_fft, err_fft, f_fft, energies, idx_top = fourier_reconstruction(f, num_coeffs_fft)
f_approx_dct, err_dct, f_dct = dct_reconstruction(f, num_coeffs_dct)

# Отображение метрик
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Ошибка Фурье", f"{err_fft:.4f}")
with col2:
    st.metric("Ошибка DCT", f"{err_dct:.4f}")
with col3:
    better_method = "Фурье" if err_fft < err_dct else "DCT"
    st.metric("Лучший метод", better_method)

# Графики восстановления
st.header("📊 Сравнение восстановления")
fig_reconstruction = plot_reconstruction_comparison(f, f_approx_fft, f_approx_dct, err_fft, err_dct)
st.plotly_chart(fig_reconstruction, use_container_width=True)

# Графики спектров
st.header("📈 Спектральный анализ")
fig_spectrum = plot_spectrum_comparison(f_fft, energies, idx_top, f_dct, num_coeffs_fft, num_coeffs_dct)
st.plotly_chart(fig_spectrum, use_container_width=True)

# Дополнительная визуализация коэффициентов
st.header("📊 Сравнение выбранных коэффициентов")
fig_coeffs = plot_coefficients_comparison(f_fft, energies, idx_top, f_dct, num_coeffs_fft, num_coeffs_dct)
st.plotly_chart(fig_coeffs, use_container_width=True)

# Таблица сравнения
st.header("📋 Детальная статистика")
stats_data = {
    'Метод': ['Фурье', 'DCT'],
    'Количество коэффициентов': [num_coeffs_fft, num_coeffs_dct],
    'Относительная ошибка': [err_fft, err_dct],
    'Средняя абсолютная ошибка': [
        np.mean(np.abs(f - f_approx_fft)),
        np.mean(np.abs(f - f_approx_dct))
    ],
    'Максимальная абсолютная ошибка': [
        np.max(np.abs(f - f_approx_fft)),
        np.max(np.abs(f - f_approx_dct))
    ]
}

df_stats = pd.DataFrame(stats_data)
st.dataframe(df_stats, use_container_width=True)

# Дополнительная информация
st.header("ℹ️ Информация о методах")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Фурье преобразование")
    st.markdown("""
    - Использует комплексные коэффициенты
    - Выбирает коэффициенты с наибольшей энергией
    - Хорошо подходит для периодических сигналов
    - Количество коэффициентов: **комплексные пары**
    """)

with col2:
    st.subheader("DCT преобразование")
    st.markdown("""
    - Использует вещественные коэффициенты
    - Берет первые коэффициенты по порядку
    - Эффективно для сигналов с плавными переходами
    - Количество коэффициентов: **вещественные числа**
    """)

# Интерактивный анализ
st.header("🔍 Интерактивный анализ")
if st.checkbox("Показать детальную информацию о коэффициентах"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Топ коэффициенты Фурье")
        top_coeffs_df = pd.DataFrame({
            'Индекс': idx_top,
            'Частота': idx_top,
            'Модуль': energies[idx_top],
            'Фаза (рад)': np.angle(f_fft[idx_top])
        })
        st.dataframe(top_coeffs_df)
    
    with col2:
        st.subheader("Первые коэффициенты DCT")
        dct_coeffs_df = pd.DataFrame({
            'Индекс': range(num_coeffs_dct),
            'Значение': f_dct[:num_coeffs_dct],
            'Модуль': np.abs(f_dct[:num_coeffs_dct])
        })
        st.dataframe(dct_coeffs_df)

st.markdown("---")
st.markdown("*Измените параметры в боковой панели для интерактивного анализа*")