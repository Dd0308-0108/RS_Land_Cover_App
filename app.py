import streamlit as st
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score
from scipy.ndimage import generic_filter
from scipy import stats

st.set_page_config(page_title="遥感影像解译工具", layout="wide")
st.title("🌍 遥感影像解译工具 APP (终极版)")

uploaded_file = st.file_uploader("上传 GeoTIFF 影像", type=["tif", "tiff"])

if uploaded_file is not None:
    with open("temp_image.tif", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with rasterio.open("temp_image.tif") as src:
        img = src.read() 
        num_bands = src.count
        img_height, img_width = src.height, src.width

    col1, col2 = st.columns([1, 3])
    with col1:
        display_mode = st.radio("选择显示模式", ["真彩色", "标准假彩色"])
        
        st.write("---")
        st.subheader("🌲 样本采集")
        class_name = st.text_input("输入当前类别名称 (如: 水体)", value="水体")
        st.info("👉 请在右侧影像上点击采集样本点！")
        
        if 'samples' not in st.session_state:
            st.session_state.samples = []
        
        if len(st.session_state.samples) > 0:
            df_samples = pd.DataFrame(st.session_state.samples)
            st.write(f"已采集样本数: {len(df_samples)}")
            csv = df_samples.to_csv(index=False).encode('utf-8')
            st.download_button("📥 下载样本数据 (CSV)", csv, "samples.csv", "text/csv")
            if st.button("🗑️ 清空所有样本"):
                st.session_state.samples = []
                st.rerun()
        else:
            st.write("暂无样本")

        st.write("---")
        st.subheader("🤖 模型训练与分类")
        
        if len(st.session_state.samples) > 0:
            unique_classes = df_samples['class'].nunique()
            if unique_classes >= 2:
                if st.button("🚀 开始随机森林分类"):
                    with st.spinner("正在训练模型、分类并执行后处理，请稍候..."):
                        feature_cols = [f"B{i+1}" for i in range(num_bands)]
                        X = df_samples[feature_cols].to_numpy(dtype=np.float64)
                        y = df_samples['class'].to_numpy(dtype=str)
                        
                        if len(df_samples) >= 5:
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                        else:
                            X_train, y_train = X, y
                            X_test, y_test = X, y 

                        rf = RandomForestClassifier(n_estimators=100, random_state=42)
                        rf.fit(X_train, y_train)

                        # 精度评价
                        y_pred_test = rf.predict(X_test)
                        oa = accuracy_score(y_test, y_pred_test)
                        kappa = cohen_kappa_score(y_test, y_pred_test)
                        
                        st.success("✅ 分类完成！")
                        st.write(f"**总体精度 (OA)**: {oa*100:.2f}%")
                        st.write(f"**Kappa 系数**: {kappa:.4f}")

                        # 全图预测
                        img_reshaped = img.reshape((num_bands, -1)).T.astype(np.float64)
                        pred_flat = rf.predict(img_reshaped)
                        pred_map = pred_flat.reshape((img_height, img_width))
                        
                        # 将字符串类别转为数字，方便画图和后处理
                        classes_list = list(rf.classes_)
                        pred_map_numeric = np.zeros(pred_map.shape, dtype=int)
                        for i, cls_name in enumerate(classes_list):
                            pred_map_numeric[pred_map == cls_name] = i
                        
                        # 🌟 自选功能：众数滤波 (3x3 窗口消除噪点)
                        # 使用 scipy 的 generic_filter 提取每个 3x3 窗口内的众数
                        filtered_map = generic_filter(pred_map_numeric, lambda x: stats.mode(x, keepdims=False)[0], size=3)
                        
                        # 保存到 session_state
                        st.session_state.pred_map_numeric = pred_map_numeric
                        st.session_state.filtered_map = filtered_map
                        st.session_state.classes = classes_list
                        
                        # 🌟 自选功能：面积统计 (假设 Landsat 像元分辨率为 30m)
                        pixel_area_ha = (30 * 30) / 10000 # 转换为公顷 (ha)
                        area_stats = {}
                        for i, cls_name in enumerate(classes_list):
                            count = np.sum(filtered_map == i)
                            area_stats[cls_name] = count * pixel_area_ha
                        st.session_state.area_stats = area_stats
            else:
                st.warning("⚠️ 请至少采集 2 个不同类别的样本才能进行分类！")
        else:
            st.write("请先在上方采集样本。")

    # 影像显示与指数计算
    if display_mode == "真彩色":
        r_band, g_band, b_band = img[2], img[1], img[0] 
    else:
        r_band, g_band, b_band = img[3], img[2], img[1] 

    def stretch_band(band):
        p2, p98 = np.percentile(band, (2, 98))
        band_clipped = np.clip(band, p2, p98)
        return (band_clipped - p2) / (p98 - p2)

    r = stretch_band(r_band)
    g = stretch_band(g_band)
    b = stretch_band(b_band)
    rgb_img = np.dstack((r, g, b))
    
    # 🌟 自选功能：NDVI (植被) 和 NDWI (水体)
    nir = img[3].astype(float)
    red = img[2].astype(float)
    green = img[1].astype(float) # 假设 B2 是绿波段
    ndvi = (nir - red) / (nir + red + 1e-10) 
    ndwi = (green - nir) / (green + nir + 1e-10)

    rgb_255 = (rgb_img * 255).astype(np.uint8)
    pil_img = Image.fromarray(rgb_255)

    with col2:
        # 增加了更多的标签页
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🖼️ 原始影像", "🗺️ 分类结果", "✨ 滤波后结果", "🌿 NDVI", "💧 NDWI 与面积"
        ])
        
        with tab1:
            display_width = 700
            clicked = streamlit_image_coordinates(pil_img, width=display_width, key="click_image")
            
            if clicked is not None:
                ratio = img_width / display_width
                real_x = int(clicked['x'] * ratio)
                real_y = int(clicked['y'] * ratio)
                
                spectra = img[:, real_y, real_x]
                sample_record = {"class": class_name, "x": real_x, "y": real_y}
                for i in range(num_bands):
                    sample_record[f"B{i+1}"] = int(spectra[i])
                
                is_duplicate = False
                if st.session_state.samples:
                    last = st.session_state.samples[-1]
                    if last["x"] == real_x and last["y"] == real_y and last["class"] == class_name:
                        is_duplicate = True
                
                if not is_duplicate:
                    st.session_state.samples.append(sample_record)

        with tab2:
            if 'pred_map_numeric' in st.session_state:
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(st.session_state.pred_map_numeric, cmap='tab20')
                ax.axis('off')
                st.pyplot(fig)
                st.write("**图例类别:**", ", ".join(st.session_state.classes))
            else:
                st.info("👈 请在左侧采集样本并分类。")

        with tab3:
            if 'filtered_map' in st.session_state:
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(st.session_state.filtered_map, cmap='tab20')
                ax.axis('off')
                st.pyplot(fig)
                st.success("✅ 已应用 3x3 众数滤波，消除了椒盐噪声。")
            else:
                st.info("👈 请在左侧采集样本并分类。")

        with tab4:
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(ndvi, cmap='RdYlGn', vmin=-0.5, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.03)
            ax.axis('off')
            st.pyplot(fig)

        with tab5:
            col_ndwi, col_area = st.columns([1, 1])
            with col_ndwi:
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(ndwi, cmap='Blues', vmin=-0.5, vmax=1)
                plt.colorbar(im, ax=ax, fraction=0.03)
                ax.axis('off')
                st.pyplot(fig)
                
            with col_area:
                st.subheader("📊 面积统计 (公顷)")
                if 'area_stats' in st.session_state:
                    # 转换为 DataFrame 展示更美观
                    df_area = pd.DataFrame(list(st.session_state.area_stats.items()), columns=['地物类别', '面积 (ha)'])
                    st.dataframe(df_area, hide_index=True)
                    st.bar_chart(df_area, x='地物类别', y='面积 (ha)')
                else:
                    st.info("请先在左侧完成分类。")