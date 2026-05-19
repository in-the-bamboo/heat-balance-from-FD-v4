import streamlit as st
import pandas as pd
import numpy as np
import trimesh
import os

# --- 1. 空間判定ロジック（L型対応・デバッグなし版） ---
def detect_rooms_from_coords(df, meshes, offset_dist=0.05):
    """座標から開口部の軸と、隣接する2つの部屋を正確に判定する"""
    if not all(col in df.columns for col in ['X[m]', 'Y[m]', 'Z[m]']):
        return None, None, None, "座標列(X[m], Y[m], Z[m])がありません"

    # 中心座標の計算
    cx, cy, cz = df['X[m]'].mean(), df['Y[m]'].mean(), df['Z[m]'].mean()

    # 判定軸の特定
    stds = {'x': df['X[m]'].std(), 'y': df['Y[m]'].std(), 'z': df['Z[m]'].std()}
    detected_axis = min(stds, key=stds.get)

    # オフセット点の計算
    pt_plus = [cx, cy, cz]
    pt_minus = [cx, cy, cz]
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[detected_axis]
    pt_plus[axis_idx] += offset_dist
    pt_minus[axis_idx] -= offset_dist

    # Ray-Casting（レイキャスト）による内外判定
    def is_inside(mesh, pt):
        min_b, max_b = mesh.bounds
        if not (min_b[0] <= pt[0] <= max_b[0] and 
                min_b[1] <= pt[1] <= max_b[1] and 
                min_b[2] <= pt[2] <= max_b[2]):
            return False
        try:
            locations, _, _ = mesh.ray.intersects_location(
                ray_origins=np.array([pt]), ray_directions=np.array([[0.0, 0.0, 1.0]])
            )
            return len(locations) % 2 == 1
        except:
            return False

    room_plus = "外気(未定義)"
    room_minus = "外気(未定義)"
    sorted_meshes = sorted(meshes.items(), key=lambda item: item[1].bounding_box.volume)

    # Plus側の判定（小さい順に調べて、見つかったらそこでストップ）
    for room_name, mesh in sorted_meshes:
        if is_inside(mesh, pt_plus): 
            room_plus = room_name
            break  # 一番小さい空間（例:エアコン）に入っていたら即確定！

    # Minus側の判定
    for room_name, mesh in sorted_meshes:
        if is_inside(mesh, pt_minus): 
            room_minus = room_name
            break
    return detected_axis, room_plus, room_minus, None


# --- 2. メインのCFDファイル一括処理関数 ---
def process_cfd_files(stl_files, cfd_files, rho, cp, lv, threshold, calc_latent, hum_col, offset_dist_m):
    # STLメッシュの読み込み（エラー文を返す既存の関数を想定）
    meshes, logs = load_stl_meshes(stl_files)
    if not meshes:
        st.error("有効な3Dメッシュ(STL)が読み込めませんでした。")
        return

    # 結果を格納するリスト
    summary_data = []

    for uploaded_file in cfd_files:
        file_name = os.path.splitext(uploaded_file.name)[0]
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, skiprows=2, encoding='cp932')
            
            # 💡 UIから指定されたオフセット距離（メートル換算後）を渡す
            axis, r_plus, r_minus, err = detect_rooms_from_coords(df, meshes, offset_dist=offset_dist_m)
            if err: continue

            # --- [既存の風量・熱量・温度の計算処理] ---
            # ※お手元のロジックで計算された数値を想定しています
            plus_flow = 150.0   # 例: ＋方向風量 [m3/h]
            plus_heat = 450.0   # 例: ＋方向熱量 [W]
            minus_flow = 20.0   # 例: ー方向風量 [m3/h]
            minus_heat = 60.0   # 例: ー方向熱量 [W]
            avg_temp = 24.5     # 例: 平均温度 [℃]
            # ----------------------------------------

            # テーブルに必要なデータを格納
            summary_data.append({
                "開口部名": file_name,
                "判定軸": axis.upper(),
                "接する部屋(+)": r_plus,
                "接する部屋(-)": r_minus,
                "平均温度 [℃]": round(avg_temp, 1),
                "風量(+) [m³/h]": round(plus_flow, 1),
                "熱量(+) [W]": round(plus_heat, 1),
                "風量(-) [m³/h]": round(minus_flow, 1),
                "熱量(-) [W]": round(minus_heat, 1)
            })
        except Exception as e:
            st.warning(f"{file_name} の処理中にエラーが発生しました: {e}")

    # 結果をデータフレーム化して画面に出力
    if summary_data:
        df_result = pd.DataFrame(summary_data)
        st.markdown("### 📊 開口部熱バランス 解析結果一覧")
        st.dataframe(df_result, use_container_width=True)
        
        # CSVダウンロードボタンもついでに配置
        csv = df_result.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 解析結果をCSVでダウンロード", csv, "heat_balance_result.csv", "text/csv")
    else:
        st.info("処理対象の開口部データがありません。")


# --- 3. Streamlit UI 画面構成 ---
st.title("FlowDesigner × Rhino 熱バランス解析ツール")

# サイドバーにUI設定を集約
st.sidebar.header("🔧 解析条件設定")

# 💡 オフセット距離をミリ単位で変更できるスライダー（初期値50mm）
offset_mm = st.sidebar.slider(
    "開口部判定のオフセット距離 (mm)", 
    min_value=10, 
    max_value=300, 
    value=50, 
    step=10,
    help="開口部の中心から表裏に何ミリ離れた点で部屋を判定するかを設定します。壁の厚みに応じて調整してください。"
)
# メートル単位に変換して計算ロジックに渡す
offset_m = offset_mm / 1000.0

# (その他の計算パラメータ UI)
rho = st.sidebar.number_input("空気密度 rho [kg/m3]", value=1.2)
cp = st.sidebar.number_input("比熱 cp [J/kg·K]", value=1006)

# ファイルアップローダー
stl_files = st.file_uploader("1. STLファイルをアップロード (複数可)", accept_multiple_files=True, type=['stl'])
cfd_files = st.file_uploader("2. 開口部CSVファイルをアップロード (複数可)", accept_multiple_files=True, type=['csv'])

if st.button("🔥 熱バランス解析を実行"):
    if stl_files and cfd_files:
        with st.spinner("空間判定および熱量計算を実行中..."):
            process_cfd_files(stl_files, cfd_files, rho, cp, None, None, False, None, offset_m)
    else:
        st.error("STLファイルと開口部CSVファイルの両方をアップロードしてください。")
