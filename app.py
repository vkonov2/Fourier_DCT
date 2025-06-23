import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.fft import rfft, irfft, dct, idct
import pandas as pd
import shapely as sp
from shapely.ops import nearest_points
from math import sqrt
import pywt

def load_cut_data(cut_name):
    """Загружает данные огранки из файла в папке rhos"""
    try:
        filename = f"rhos/{cut_name}.txt"
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

def load_angles_data(cut_name):
    """Загружает углы огранки из файла в папке angles"""
    try:
        filename = f"angles/{cut_name}.txt"
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
        st.warning(f"Файл углов {filename} не найден! Используются равномерно распределенные углы.")
        return None

def get_available_cuts():
    """Возвращает список доступных огранок"""
    import os
    cuts = []
    if os.path.exists('rhos'):
        for file in os.listdir('rhos'):
            if file.endswith('.txt'):
                cuts.append(file.replace('.txt', ''))
    return sorted(cuts)

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
    
    # Выбираем коэффициенты с наибольшими значениями
    dct_abs = np.abs(f_dct)
    idx_top = np.argsort(dct_abs)[-num_coeffs:][::-1]
    
    f_dct_approx = np.zeros_like(f_dct)
    f_dct_approx[idx_top] = f_dct[idx_top]
    f_approx = idct(f_dct_approx, type=2, norm='ortho')
    
    err = np.linalg.norm(f - f_approx) / np.linalg.norm(f)
    return f_approx, err, f_dct, idx_top

def wavelet_reconstruction(f, f_approx, wavelet_type, wavelet_n):
    """Восстановление разности с помощью вейвлетов"""
    diff = f - f_approx
    
    try:
        coeffs = pywt.wavedec(diff, wavelet_type, level=None, mode="periodization")
    except Exception as e:
        print(f"Ошибка pywt.wavedec: {e}")
        return f_approx, float('nan'), None
    
    # Сглаживаем коэффициенты
    coeffs_flat = [(abs(c), i, j) for i, level in enumerate(coeffs) for j, c in enumerate(level)]
    coeffs_flat.sort(reverse=True)
    
    if wavelet_n > len(coeffs_flat):
        print(f"Предупреждение: wavelet_n слишком много: {wavelet_n} > {len(coeffs_flat)}")
        return f_approx, float('nan'), None
    
    new_coeffs = [np.zeros_like(level) for level in coeffs]
    for _, i, j in coeffs_flat[:wavelet_n]:
        new_coeffs[i][j] = coeffs[i][j]
    
    try:
        wavelet_diff = pywt.waverec(new_coeffs, wavelet_type, mode="periodization")
    except Exception as e:
        print(f"Ошибка pywt.waverec: {e}")
        return f_approx, float('nan'), None
    
    if len(wavelet_diff) != len(f) or np.any(np.isnan(wavelet_diff)) or np.any(np.isinf(wavelet_diff)):
        print("Предупреждение: wavelet_diff содержит ошибки")
        return f_approx, float('nan'), None
    
    result = f_approx + wavelet_diff
    
    if np.any(np.isnan(result)) or np.any(np.isinf(result)):
        print("Предупреждение: результат содержит nan/inf")
        return f_approx, float('nan'), None
    
    err = np.linalg.norm(f - result) / np.linalg.norm(f)
    return result, err, wavelet_diff

def fourier_wavelet_reconstruction(f, num_coeffs_fft, wavelet_type, wavelet_n):
    """Комбинированное восстановление: Фурье + вейвлеты"""
    f_approx_fft, _, f_fft, energies, idx_top = fourier_reconstruction(f, num_coeffs_fft)
    f_approx_combined, err_combined, wavelet_diff = wavelet_reconstruction(f, f_approx_fft, wavelet_type, wavelet_n)
    return f_approx_combined, err_combined, f_approx_fft, wavelet_diff

def dct_wavelet_reconstruction(f, num_coeffs_dct, wavelet_type, wavelet_n):
    """Комбинированное восстановление: DCT + вейвлеты"""
    f_approx_dct, _, f_dct, idx_top_dct = dct_reconstruction(f, num_coeffs_dct)
    f_approx_combined, err_combined, wavelet_diff = wavelet_reconstruction(f, f_approx_dct, wavelet_type, wavelet_n)
    return f_approx_combined, err_combined, f_approx_dct, wavelet_diff

def plot_reconstruction_comparison(f, f_approx_fft, f_approx_dct, f_approx_fft_wave, f_approx_dct_wave, 
                                 err_fft, err_dct, err_fft_wave, err_dct_wave):
    """Построение графиков сравнения восстановления с помощью Plotly"""
    x = np.arange(len(f))
    
    # Создаем подграфики
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Сравнение восстановления сигнала', 'Ошибки восстановления (со знаками)'),
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
    fig.add_trace(
        go.Scatter(x=x, y=f_approx_fft_wave, mode='lines', 
                  name=f'Фурье+Вейвлеты (ошибка: {err_fft_wave:.4f})',
                  line=dict(color='orange', width=2, dash='dot')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x, y=f_approx_dct_wave, mode='lines', 
                  name=f'DCT+Вейвлеты (ошибка: {err_dct_wave:.4f})',
                  line=dict(color='purple', width=2, dash='dot')),
        row=1, col=1
    )
    
    # График ошибок восстановления
    error_fft = f - f_approx_fft
    error_dct = f - f_approx_dct
    error_fft_wave = f - f_approx_fft_wave
    error_dct_wave = f - f_approx_dct_wave
    
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
    fig.add_trace(
        go.Scatter(x=x, y=error_fft_wave, mode='lines', 
                  name=f'Ошибка Фурье+Вейвлеты (средняя: {np.mean(error_fft_wave):.4f})',
                  line=dict(color='orange', width=1)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=x, y=error_dct_wave, mode='lines', 
                  name=f'Ошибка DCT+Вейвлеты (средняя: {np.mean(error_dct_wave):.4f})',
                  line=dict(color='purple', width=1)),
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
    fig.update_yaxes(title_text="Ошибка (со знаком)", row=2, col=1)
    
    return fig

def plot_spectrum_comparison(f_fft, energies, idx_top, f_dct, idx_top_dct, num_coeffs_fft, num_coeffs_dct):
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
        go.Scatter(x=dct_coeffs[idx_top_dct], y=np.abs(f_dct[idx_top_dct]), 
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

def plot_coefficients_comparison(f_fft, energies, idx_top, f_dct, idx_top_dct, num_coeffs_fft, num_coeffs_dct):
    """Дополнительная визуализация сравнения коэффициентов"""
    # Создаем DataFrame для визуализации
    fourier_data = pd.DataFrame({
        'Индекс': idx_top,
        'Модуль': energies[idx_top],
        'Метод': 'Фурье'
    })
    
    dct_data = pd.DataFrame({
        'Индекс': idx_top_dct,
        'Модуль': np.abs(f_dct[idx_top_dct]),
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

def plot_wavelet_errors(f, f_approx_fft, f_approx_dct, wavelet_diff_fft, wavelet_diff_dct):
    """Визуализация ошибок восстановления вейвлетами"""
    try:
        x = np.arange(len(f))
        
        # Вычисляем разности
        diff_fft = f - f_approx_fft
        diff_dct = f - f_approx_dct
        
        # Создаем подграфики
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Ошибка восстановления Фурье vs DCT', 'Ошибка восстановления вейвлетами Фурье vs DCT'),
            vertical_spacing=0.1
        )
        
        # График ошибок восстановления Фурье vs DCT
        fig.add_trace(
            go.Scatter(x=x, y=diff_fft, mode='lines', 
                      name=f'Ошибка Фурье (средняя: {np.mean(diff_fft):.4f})',
                      line=dict(color='red', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=diff_dct, mode='lines', 
                      name=f'Ошибка DCT (средняя: {np.mean(diff_dct):.4f})',
                      line=dict(color='green', width=2)),
            row=1, col=1
        )
        
        # График ошибок восстановления вейвлетами
        if wavelet_diff_fft is not None:
            wavelet_error_fft = diff_fft - wavelet_diff_fft
            fig.add_trace(
                go.Scatter(x=x, y=wavelet_error_fft, mode='lines', 
                          name=f'Ошибка вейвлетов Фурье (средняя: {np.mean(wavelet_error_fft):.4f})',
                          line=dict(color='orange', width=2)),
                row=2, col=1
            )
        
        if wavelet_diff_dct is not None:
            wavelet_error_dct = diff_dct - wavelet_diff_dct
            fig.add_trace(
                go.Scatter(x=x, y=wavelet_error_dct, mode='lines', 
                          name=f'Ошибка вейвлетов DCT (средняя: {np.mean(wavelet_error_dct):.4f})',
                          line=dict(color='purple', width=2)),
                row=2, col=1
            )
        
        # Настройка макета
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Анализ ошибок восстановления",
            title_x=0.5
        )
        
        # Настройка осей
        fig.update_xaxes(title_text="Индекс", row=1, col=1)
        fig.update_yaxes(title_text="Ошибка", row=1, col=1)
        fig.update_xaxes(title_text="Индекс", row=2, col=1)
        fig.update_yaxes(title_text="Ошибка вейвлетов", row=2, col=1)
        
        return fig
    except Exception as e:
        st.error(f"Ошибка при создании графика ошибок: {e}")
        # Возвращаем пустой график
        fig = go.Figure()
        fig.add_annotation(
            text="Ошибка при создании графика ошибок",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(height=400)
        return fig

def plot_contours_comparison(f, f_approx_fft, f_approx_dct, f_approx_fft_wave, f_approx_dct_wave, angles):
    """Визуализация сравнения контуров"""
    try:
        # Создаем точки для каждого контура
        original_pts = recover_points_from_rhos(f, angles)
        fft_pts = recover_points_from_rhos(f_approx_fft, angles)
        dct_pts = recover_points_from_rhos(f_approx_dct, angles)
        fft_wave_pts = recover_points_from_rhos(f_approx_fft_wave, angles)
        dct_wave_pts = recover_points_from_rhos(f_approx_dct_wave, angles)
        
        # Создаем контуры
        original_contour = sp.LinearRing(original_pts)
        fft_contour = sp.LinearRing(fft_pts)
        dct_contour = sp.LinearRing(dct_pts)
        fft_wave_contour = sp.LinearRing(fft_wave_pts)
        dct_wave_contour = sp.LinearRing(dct_wave_pts)
        
        # Вычисляем метрики
        fft_metrics = get_metrics(original_contour, fft_contour)
        dct_metrics = get_metrics(original_contour, dct_contour)
        fft_wave_metrics = get_metrics(original_contour, fft_wave_contour)
        dct_wave_metrics = get_metrics(original_contour, dct_wave_contour)
        
        # Создаем график
        fig = go.Figure()
        
        # Добавляем контуры
        fig.add_trace(go.Scatter(
            x=original_pts[:, 0], y=original_pts[:, 1],
            mode='lines', name='Исходный контур',
            line=dict(color='blue', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=fft_pts[:, 0], y=fft_pts[:, 1],
            mode='lines', name=f'Фурье (maxQ: {fft_metrics[2]:.2f}%, avgQ: {fft_metrics[3]:.2f}%)',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=dct_pts[:, 0], y=dct_pts[:, 1],
            mode='lines', name=f'DCT (maxQ: {dct_metrics[2]:.2f}%, avgQ: {dct_metrics[3]:.2f}%)',
            line=dict(color='green', width=2, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=fft_wave_pts[:, 0], y=fft_wave_pts[:, 1],
            mode='lines', name=f'Фурье+Вейвлеты (maxQ: {fft_wave_metrics[2]:.2f}%, avgQ: {fft_wave_metrics[3]:.2f}%)',
            line=dict(color='orange', width=2, dash='dot')
        ))
        
        fig.add_trace(go.Scatter(
            x=dct_wave_pts[:, 0], y=dct_wave_pts[:, 1],
            mode='lines', name=f'DCT+Вейвлеты (maxQ: {dct_wave_metrics[2]:.2f}%, avgQ: {dct_wave_metrics[3]:.2f}%)',
            line=dict(color='purple', width=2, dash='dot')
        ))
        
        # Настройка макета
        fig.update_layout(
            title="Сравнение контуров",
            xaxis_title="X",
            yaxis_title="Y",
            height=600,
            showlegend=True,
            title_x=0.5
        )
        
        # Устанавливаем равные масштабы осей
        fig.update_xaxes(scaleanchor="y", scaleratio=1)
        
        return fig, {
            'Фурье': fft_metrics,
            'DCT': dct_metrics,
            'Фурье+Вейвлеты': fft_wave_metrics,
            'DCT+Вейвлеты': dct_wave_metrics
        }
    except Exception as e:
        st.error(f"Ошибка при создании контуров: {e}")
        # Возвращаем пустой график и пустые метрики
        fig = go.Figure()
        fig.add_annotation(
            text="Ошибка при создании контуров",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(height=400)
        
        empty_metrics = {
            'Фурье': [float('nan'), float('nan'), float('nan'), float('nan')],
            'DCT': [float('nan'), float('nan'), float('nan'), float('nan')],
            'Фурье+Вейвлеты': [float('nan'), float('nan'), float('nan'), float('nan')],
            'DCT+Вейвлеты': [float('nan'), float('nan'), float('nan'), float('nan')]
        }
        
        return fig, empty_metrics

def get_metrics(reference_geom, geom):
    max_diameter = sqrt((reference_geom.bounds[2]-reference_geom.bounds[0])**2+(reference_geom.bounds[3]-reference_geom.bounds[1])**2)
    reference_points = [sp.get_point(reference_geom, idx) for idx in range(sp.get_num_points(reference_geom))]
    nearest_points_on_geom = [nearest_points(geom, point)[0] for point in reference_points]
    distances = [point.distance(nearest_point) for point, nearest_point in zip(reference_points, nearest_points_on_geom)]
    max_dist, avg_dist = max(distances), sum(distances) / len(distances)

    geom_points = [sp.get_point(geom, idx) for idx in range(sp.get_num_points(geom))]
    nearest_points_on_ref = [nearest_points(reference_geom, point)[0] for point in geom_points]
    distances_ref = [point.distance(nearest_point) for point, nearest_point in zip(geom_points, nearest_points_on_ref)]
    max_dist_ref, avg_dist_ref = max(distances_ref), sum(distances_ref) / len(distances_ref)

    max_distance = max(max_dist, max_dist_ref)
    avg_distance = (avg_dist + avg_dist_ref) / 2
    max_quality = max_distance / max_diameter * 100
    avg_quality = avg_distance / max_diameter * 100
    return max_distance, avg_distance, max_quality, avg_quality

def recover_points_from_rhos(rhos, angles, cut=None, logger=None):
    if np.any(np.isnan(rhos)) or np.any(np.isinf(rhos)):
        if logger: 
            logger.warning("recoverPointsFromRhos вход nan/inf")
        else:
            print(f"[{cut}] recoverPointsFromRhos вход nan/inf")
        return None
    return np.stack((rhos * np.cos(angles), rhos * np.sin(angles)), axis=-1)


# Настройка страницы Streamlit
st.set_page_config(page_title="Визуализация восстановления сигнала", layout="wide")

# Динамический заголовок
if 'selected_cut' in st.session_state:
    st.title(f"💎 Визуализация восстановления сигнала: {st.session_state.selected_cut} (Фурье vs DCT)")
else:
    st.title("🎯 Визуализация восстановления сигнала: Фурье vs DCT")

# Загрузка данных
st.info("📋 **Настройки находятся в боковой панели слева!** ")

# Боковая панель с настройками
st.sidebar.header("⚙️ Настройки параметров")

# Выбор огранки
available_cuts = get_available_cuts()
if available_cuts:
    selected_cut = st.sidebar.selectbox(
        "Выберите огранку",
        options=available_cuts,
        index=available_cuts.index('Pear') if 'Pear' in available_cuts else 0,
        help="Выберите тип огранки для анализа"
    )
    
    # Сохраняем выбранную огранку в session_state
    st.session_state.selected_cut = selected_cut
    
    # Загрузка данных выбранной огранки
    f = load_cut_data(selected_cut)
    if f is None:
        st.error("Ошибка при загрузке данных огранки")
        st.stop()
    
    # Загрузка углов для выбранной огранки
    angles = load_angles_data(selected_cut)
    if angles is None:
        # Если файл углов не найден, используем равномерно распределенные углы
        angles = np.linspace(0, 2*np.pi, len(f), endpoint=False)
        st.info(f"ℹ️ Используются равномерно распределенные углы для {len(f)} точек")
    else:
        # Проверяем соответствие количества углов и точек
        if len(angles) != len(f):
            st.warning(f"⚠️ Количество углов ({len(angles)}) не соответствует количеству точек ({len(f)}). Используются равномерно распределенные углы.")
            angles = np.linspace(0, 2*np.pi, len(f), endpoint=False)
    
    # st.success(f"✅ Загружена огранка: **{selected_cut}** ({len(f)} точек)")
else:
    st.error("Ошибка при загрузке данных")
    st.stop()

# Слайдеры для количества коэффициентов
max_coeffs_fft = min(50, len(f) // 2)  # Максимум для Фурье
max_coeffs_dct = min(100, len(f))      # Максимум для DCT

# Инициализация session_state
if 'num_coeffs_fft' not in st.session_state:
    st.session_state.num_coeffs_fft = 12
if 'num_coeffs_dct' not in st.session_state:
    st.session_state.num_coeffs_dct = 24
if 'dct_fixed_ratio' not in st.session_state:
    st.session_state.dct_fixed_ratio = False
if 'wavelet_n' not in st.session_state:
    st.session_state.wavelet_n = 8

# Функции для обновления значений
def update_fft_slider():
    st.session_state.num_coeffs_fft = st.session_state.fft_slider
    if st.session_state.dct_fixed_ratio:
        st.session_state.num_coeffs_dct = st.session_state.num_coeffs_fft * 2
        # Принудительно обновляем элементы управления DCT
        st.session_state.dct_slider = st.session_state.num_coeffs_dct
        st.session_state.dct_input = st.session_state.num_coeffs_dct

def update_fft_input():
    st.session_state.num_coeffs_fft = st.session_state.fft_input
    if st.session_state.dct_fixed_ratio:
        st.session_state.num_coeffs_dct = st.session_state.num_coeffs_fft * 2
        # Принудительно обновляем элементы управления DCT
        st.session_state.dct_slider = st.session_state.num_coeffs_dct
        st.session_state.dct_input = st.session_state.num_coeffs_dct

def update_dct_slider():
    if not st.session_state.dct_fixed_ratio:
        st.session_state.num_coeffs_dct = st.session_state.dct_slider

def update_dct_input():
    if not st.session_state.dct_fixed_ratio:
        st.session_state.num_coeffs_dct = st.session_state.dct_input

def update_fixed_ratio():
    if st.session_state.dct_fixed_ratio:
        st.session_state.num_coeffs_dct = st.session_state.num_coeffs_fft * 2
        # Принудительно обновляем элементы управления DCT
        st.session_state.dct_slider = st.session_state.num_coeffs_dct
        st.session_state.dct_input = st.session_state.num_coeffs_dct

def update_wavelet_slider():
    st.session_state.wavelet_n = st.session_state.wavelet_slider
    # Принудительно обновляем поле ввода
    st.session_state.wavelet_input = st.session_state.wavelet_n

def update_wavelet_input():
    st.session_state.wavelet_n = st.session_state.wavelet_input
    # Принудительно обновляем слайдер
    st.session_state.wavelet_slider = st.session_state.wavelet_n

# Галочка для фиксации соотношения
dct_fixed_ratio = st.sidebar.checkbox(
    "Фиксировать DCT = 2 × Фурье",
    value=st.session_state.dct_fixed_ratio,
    key="dct_fixed_ratio",
    on_change=update_fixed_ratio,
    help="Если включено, количество коэффициентов DCT будет автоматически в 2 раза больше коэффициентов Фурье"
)

# Слайдер для коэффициентов Фурье
num_coeffs_fft = st.sidebar.slider(
    "Количество коэффициентов Фурье", 
    min_value=1, 
    max_value=max_coeffs_fft, 
    value=st.session_state.num_coeffs_fft,
    key="fft_slider",
    on_change=update_fft_slider,
    help="Количество самых значимых комплексных коэффициентов Фурье"
)

# Текстовое поле для ручного ввода коэффициентов Фурье
num_coeffs_fft_input = st.sidebar.number_input(
    "Введите количество коэффициентов Фурье",
    min_value=1,
    max_value=max_coeffs_fft,
    value=st.session_state.num_coeffs_fft,
    key="fft_input",
    on_change=update_fft_input,
    help="Введите точное количество коэффициентов Фурье"
)

# Слайдер для коэффициентов DCT (отключается при фиксированном соотношении)
num_coeffs_dct = st.sidebar.slider(
    "Количество коэффициентов DCT", 
    min_value=1, 
    max_value=max_coeffs_dct, 
    value=st.session_state.num_coeffs_dct,
    key="dct_slider",
    on_change=update_dct_slider,
    disabled=dct_fixed_ratio,
    help="Количество первых коэффициентов DCT" + (" (автоматически)" if dct_fixed_ratio else "")
)

# Текстовое поле для ручного ввода коэффициентов DCT (отключается при фиксированном соотношении)
num_coeffs_dct_input = st.sidebar.number_input(
    "Введите количество коэффициентов DCT",
    min_value=1,
    max_value=max_coeffs_dct,
    value=st.session_state.num_coeffs_dct,
    key="dct_input",
    on_change=update_dct_input,
    disabled=dct_fixed_ratio,
    help="Введите точное количество коэффициентов DCT" + (" (автоматически)" if dct_fixed_ratio else "")
)

# Параметры вейвлетов
st.sidebar.header("🌊 Параметры вейвлетов")

# Выбор типа вейвлета
wavelet_types = [
        'bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4',
        'bior5.5', 'bior6.8', 'coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'coif6', 'coif7', 'coif8', 'coif9', 'coif10', 'coif11', 'coif12', 'coif13', 'coif14', 'coif15', 'coif16', 'coif17',
        'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11', 'db12', 'db13', 'db14', 'db15', 'db16', 'db17', 'db18', 'db19', 'db20', 'db21', 'db22', 'db23', 'db24', 'db25', 'db26', 'db27', 'db28', 'db29', 'db30', 'db31', 'db32', 'db33', 'db34', 'db35', 'db36', 'db37', 'db38', 'dmey', 'haar',
        'rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4', 'rbio2.6', 'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5', 'rbio3.7', 'rbio3.9', 'rbio4.4', 'rbio5.5', 'rbio6.8',
        'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 'sym9', 'sym10', 'sym11', 'sym12', 'sym13', 'sym14', 'sym15', 'sym16', 'sym17', 'sym18', 'sym19', 'sym20'
    ]

wavelet_type = st.sidebar.selectbox(
    "Тип вейвлета",
    options=wavelet_types,
    index=wavelet_types.index('sym10'),
    help="Выберите тип вейвлета для аппроксимации разности"
)

# Количество коэффициентов вейвлетов
max_wavelet_coeffs = 50
wavelet_n = st.sidebar.slider(
    "Количество коэффициентов вейвлетов (WaveN)",
    min_value=1,
    max_value=max_wavelet_coeffs,
    value=st.session_state.wavelet_n,
    key="wavelet_slider",
    on_change=update_wavelet_slider,
    help="Количество наибольших коэффициентов вейвлетов для аппроксимации разности"
)

# Текстовое поле для ручного ввода коэффициентов вейвлетов
wavelet_n_input = st.sidebar.number_input(
    "Введите количество коэффициентов вейвлетов",
    min_value=1,
    max_value=max_wavelet_coeffs,
    value=st.session_state.wavelet_n,
    key="wavelet_input",
    on_change=update_wavelet_input,
    help="Введите точное количество коэффициентов вейвлетов"
)

# Выполнение восстановления
f_approx_fft, err_fft, f_fft, energies, idx_top = fourier_reconstruction(f, num_coeffs_fft_input)
f_approx_dct, err_dct, f_dct, idx_top_dct = dct_reconstruction(f, num_coeffs_dct_input)

# Восстановление с вейвлетами
f_approx_fft_wave, err_fft_wave, f_approx_fft_only, wavelet_diff_fft = fourier_wavelet_reconstruction(f, num_coeffs_fft_input, wavelet_type, wavelet_n_input)
f_approx_dct_wave, err_dct_wave, f_approx_dct_only, wavelet_diff_dct = dct_wavelet_reconstruction(f, num_coeffs_dct_input, wavelet_type, wavelet_n_input)

# Обработка ошибок вейвлетов
if np.isnan(err_fft_wave):
    f_approx_fft_wave = f_approx_fft
    err_fft_wave = err_fft
    wavelet_diff_fft = None
    st.warning("⚠️ Восстановление Фурье+Вейвлеты не удалось, используется только Фурье")

if np.isnan(err_dct_wave):
    f_approx_dct_wave = f_approx_dct
    err_dct_wave = err_dct
    wavelet_diff_dct = None
    st.warning("⚠️ Восстановление DCT+Вейвлеты не удалось, используется только DCT")

# Отображение метрик
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Ошибка Фурье", f"{err_fft:.4f}")
with col2:
    st.metric("Ошибка DCT", f"{err_dct:.4f}")
with col3:
    st.metric("Ошибка Фурье+Вейвлеты", f"{err_fft_wave:.4f}")
with col4:
    st.metric("Ошибка DCT+Вейвлеты", f"{err_dct_wave:.4f}")
with col5:
    methods = ["Фурье", "DCT", "Фурье+Вейвлеты", "DCT+Вейвлеты"]
    errors = [err_fft, err_dct, err_fft_wave, err_dct_wave]
    best_method = methods[np.argmin(errors)]
    st.metric("Лучший метод", best_method)

# Графики восстановления
st.header("📊 Сравнение восстановления")
fig_reconstruction = plot_reconstruction_comparison(f, f_approx_fft, f_approx_dct, f_approx_fft_wave, f_approx_dct_wave, 
                                                   err_fft, err_dct, err_fft_wave, err_dct_wave)
st.plotly_chart(fig_reconstruction, use_container_width=True)

# Графики ошибок восстановления
st.header("📊 Анализ ошибок восстановления")
fig_wavelet_errors = plot_wavelet_errors(f, f_approx_fft, f_approx_dct, wavelet_diff_fft, wavelet_diff_dct)
st.plotly_chart(fig_wavelet_errors, use_container_width=True)

# Визуализация контуров
st.header("🎯 Сравнение контуров")
fig_contours, contour_metrics = plot_contours_comparison(f, f_approx_fft, f_approx_dct, f_approx_fft_wave, f_approx_dct_wave, angles)
st.plotly_chart(fig_contours, use_container_width=True)

# Таблица метрик контуров
st.subheader("📋 Метрики качества контуров")
contour_stats_data = {
    'Метод': ['Фурье', 'DCT', 'Фурье+Вейвлеты', 'DCT+Вейвлеты'],
    'maxD': [
        f"{contour_metrics['Фурье'][0]:.6f}" if not np.isnan(contour_metrics['Фурье'][0]) else "N/A",
        f"{contour_metrics['DCT'][0]:.6f}" if not np.isnan(contour_metrics['DCT'][0]) else "N/A",
        f"{contour_metrics['Фурье+Вейвлеты'][0]:.6f}" if not np.isnan(contour_metrics['Фурье+Вейвлеты'][0]) else "N/A",
        f"{contour_metrics['DCT+Вейвлеты'][0]:.6f}" if not np.isnan(contour_metrics['DCT+Вейвлеты'][0]) else "N/A"
    ],
    'avgD': [
        f"{contour_metrics['Фурье'][1]:.6f}" if not np.isnan(contour_metrics['Фурье'][1]) else "N/A",
        f"{contour_metrics['DCT'][1]:.6f}" if not np.isnan(contour_metrics['DCT'][1]) else "N/A",
        f"{contour_metrics['Фурье+Вейвлеты'][1]:.6f}" if not np.isnan(contour_metrics['Фурье+Вейвлеты'][1]) else "N/A",
        f"{contour_metrics['DCT+Вейвлеты'][1]:.6f}" if not np.isnan(contour_metrics['DCT+Вейвлеты'][1]) else "N/A"
    ],
    'maxQ': [
        f"{contour_metrics['Фурье'][2]:.6f}" if not np.isnan(contour_metrics['Фурье'][2]) else "N/A",
        f"{contour_metrics['DCT'][2]:.6f}" if not np.isnan(contour_metrics['DCT'][2]) else "N/A",
        f"{contour_metrics['Фурье+Вейвлеты'][2]:.6f}" if not np.isnan(contour_metrics['Фурье+Вейвлеты'][2]) else "N/A",
        f"{contour_metrics['DCT+Вейвлеты'][2]:.6f}" if not np.isnan(contour_metrics['DCT+Вейвлеты'][2]) else "N/A"
    ],
    'avgQ': [
        f"{contour_metrics['Фурье'][3]:.6f}" if not np.isnan(contour_metrics['Фурье'][3]) else "N/A",
        f"{contour_metrics['DCT'][3]:.6f}" if not np.isnan(contour_metrics['DCT'][3]) else "N/A",
        f"{contour_metrics['Фурье+Вейвлеты'][3]:.6f}" if not np.isnan(contour_metrics['Фурье+Вейвлеты'][3]) else "N/A",
        f"{contour_metrics['DCT+Вейвлеты'][3]:.6f}" if not np.isnan(contour_metrics['DCT+Вейвлеты'][3]) else "N/A"
    ]
}

df_contour_stats = pd.DataFrame(contour_stats_data)
st.dataframe(df_contour_stats, use_container_width=True)

# Графики спектров
st.header("📈 Спектральный анализ")
fig_spectrum = plot_spectrum_comparison(f_fft, energies, idx_top, f_dct, idx_top_dct, num_coeffs_fft_input, num_coeffs_dct_input)
st.plotly_chart(fig_spectrum, use_container_width=True)

# Дополнительная визуализация коэффициентов
st.header("📊 Сравнение выбранных коэффициентов")
fig_coeffs = plot_coefficients_comparison(f_fft, energies, idx_top, f_dct, idx_top_dct, num_coeffs_fft_input, num_coeffs_dct_input)
st.plotly_chart(fig_coeffs, use_container_width=True)

# Таблица сравнения
st.header("📋 Детальная статистика")
stats_data = {
    'Метод': ['Фурье', 'DCT', 'Фурье+Вейвлеты', 'DCT+Вейвлеты'],
    'Коэф. Фурье': [num_coeffs_fft_input, '-', num_coeffs_fft_input, '-'],
    'Коэф. DCT': ['-', num_coeffs_dct_input, '-', num_coeffs_dct_input],
    'Коэф. Вейвлетов': ['-', '-', wavelet_n_input, wavelet_n_input],
    'Относительная ошибка': [err_fft, err_dct, err_fft_wave, err_dct_wave],
    'Средняя абсолютная ошибка': [
        np.mean(np.abs(f - f_approx_fft)),
        np.mean(np.abs(f - f_approx_dct)),
        np.mean(np.abs(f - f_approx_fft_wave)),
        np.mean(np.abs(f - f_approx_dct_wave))
    ],
    'Максимальная абсолютная ошибка': [
        np.max(np.abs(f - f_approx_fft)),
        np.max(np.abs(f - f_approx_dct)),
        np.max(np.abs(f - f_approx_fft_wave)),
        np.max(np.abs(f - f_approx_dct_wave))
    ]
}

df_stats = pd.DataFrame(stats_data)
st.dataframe(df_stats, use_container_width=True)

# Дополнительная информация
st.header("ℹ️ Информация о методах")
col1, col2, col3 = st.columns(3)

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

with col3:
    st.subheader("Вейвлет преобразование")
    st.markdown("""
    - Многоуровневое разложение сигнала
    - Эффективно для аппроксимации разности
    - Выбирает наибольшие коэффициенты
    - Тип вейвлета: **{wavelet_type}**
    - Количество коэффициентов: **{wavelet_n_input}**
    """.format(wavelet_type=wavelet_type, wavelet_n_input=wavelet_n_input))

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
        st.dataframe(top_coeffs_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("Выбранные коэффициенты DCT")
        dct_coeffs_df = pd.DataFrame({
            'Индекс': idx_top_dct,
            'Значение': f_dct[idx_top_dct],
            'Модуль': np.abs(f_dct[idx_top_dct])
        })
        st.dataframe(dct_coeffs_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("*Измените параметры в боковой панели для интерактивного анализа*")