import streamlit as st
import pandas as pd
import os
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import matplotlib_fontja
import numpy as np
import trimesh  # 3Dメッシュの内外判定用ライブラリ

# ==========================================
# 1. 関数定義
# ==========================================

def process_cfd_files_with_stl(stl_files, cfd_files, rho, cp, threshold, offset_dist, vent_settings):
    """
    STLファイル（部屋の立体データ）を使って、開口部CSVがどの部屋を繋いでいるかを自動判定する
    """
    logs = []
    room_meshes = {}

    # --- 1. 部屋のSTLファイルを読み込んで辞書化 ---
    if not stl_files:
        return None, None, None, ["❌ STLファイルがアップロードされていません。"]
    
    for stl_file in stl_files:
        try:
            room_name = os.path.splitext(stl_file.name)[0]
            # バイナリとして読み込み
            stl_data = io.BytesIO(stl_file.read())
            mesh = trimesh.load(stl_data, file_type='stl')
            room_meshes[room_name] = mesh
            logs.append(f"📁 STL読み込み成功: {room_name}")
        except Exception as e:
            logs.append(f"❌ STL読み込み失敗: {stl_file.name} ({e})")

    opening_results_list = []
    files_to_process = []
    
    for f in cfd_files:
        files_to_process.append({'file': f, 'type': 'normal'})
    
    # 換気CSVがアップロードされていればリストに合流させる
    if vent_settings.get('in_file') is not None:
        files_to_process.append({
            'file': vent_settings['in_file'], 
            'type': 'vent_in', 
            'target_room': vent_settings['in_room']
        })
    if vent_settings.get('out_file') is not None:
        files_to_process.append({
            'file': vent_settings['out_file'], 
            'type': 'vent_out', 
            'target_room': vent_settings['out_room']
        })
    

    # --- 2. 各CFDファイルをループ処理 ---
    total_files = len(files_to_process)
    progress_bar = st.progress(0)

    for i, file_info in enumerate(files_to_process):
        progress_bar.progress((i + 1) / total_files)

        uploaded_file = file_info['file']
        file_name = uploaded_file.name
        file_key = os.path.splitext(file_name)[0]
        detected_axis = None

        # --- (A) 全データと軸情報の読み込み ---
        try:
            uploaded_file.seek(0)
            # 軸判定用に先頭部分をチェック
            df_temp = pd.read_csv(uploaded_file, skiprows=2, nrows=1, encoding='cp932')
            axis_col = [c for c in df_temp.columns if '流量算出面' in str(c)]

            if axis_col:
                raw_axis_value = str(df_temp[axis_col[0]].iloc[0]).strip()
                if raw_axis_value:
                    detected_axis = raw_axis_value[0].lower() # X, Y, Z -> x, y, z
            else:
                logs.append(f"⚠️ {file_name}: '流量算出面' 列が見つからないためスキップします。")
                continue

            # 本データの読み込み
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, skiprows=2, encoding='cp932')
            
            # 座標列の抽出
            x_col = [c for c in df.columns if 'X[m]' in str(c) or c == 'X']
            y_col = [c for c in df.columns if 'Y[m]' in str(c) or c == 'Y']
            z_col = [c for c in df.columns if 'Z[m]' in str(c) or c == 'Z']

            if not (x_col and y_col and z_col):
                logs.append(f"⚠️ {file_name}: 座標列(X[m], Y[m], Z[m])が見つかりません。")
                continue

            # 数値変換と欠損値削除
            flow_col, temp_col = '流量[m3/h]', 'スカラー量[℃]'
            df[flow_col] = pd.to_numeric(df[flow_col], errors='coerce')
            df[temp_col] = pd.to_numeric(df[temp_col], errors='coerce')
            df[x_col[0]] = pd.to_numeric(df[x_col[0]], errors='coerce')
            df[y_col[0]] = pd.to_numeric(df[y_col[0]], errors='coerce')
            df[z_col[0]] = pd.to_numeric(df[z_col[0]], errors='coerce')
            df.dropna(subset=[flow_col, temp_col, x_col[0], y_col[0], z_col[0]], inplace=True)

        except Exception as e:
            logs.append(f"❌ CSV読み込みエラー: {file_name} ({e})")
            continue

        # --- (B) 中心点・法線の計算とSTLによる部屋判定 ---
        if file_info['type'] == 'normal':
            try:
                center_pt = np.array([df[x_col[0]].mean(), df[y_col[0]].mean(), df[z_col[0]].mean()])
                normal_vec = np.array([0.0, 0.0, 0.0])
                if detected_axis == 'x':   normal_vec = np.array([1.0, 0.0, 0.0])
                elif detected_axis == 'y': normal_vec = np.array([0.0, 1.0, 0.0])
                elif detected_axis == 'z': normal_vec = np.array([0.0, 0.0, 1.0])

                probe_plus = center_pt + (normal_vec * offset_dist)
                probe_minus = center_pt - (normal_vec * offset_dist)

                plus_candidates = []
                minus_candidates = []

                for room_name, mesh in room_meshes.items():
                    try:
                        vol = abs(mesh.volume)
                        if vol < 1e-5: vol = mesh.bounding_box.volume
                    except:
                        vol = mesh.bounding_box.volume

                    if mesh.contains([probe_plus])[0]: plus_candidates.append((room_name, vol))
                    if mesh.contains([probe_minus])[0]: minus_candidates.append((room_name, vol))

                found_plus_room = min(plus_candidates, key=lambda x: x[1])[0] if plus_candidates else "外部(未特定)"
                found_minus_room = min(minus_candidates, key=lambda x: x[1])[0] if minus_candidates else "外部(未特定)"

                if found_plus_room == "外部(未特定)" and found_minus_room == "外部(未特定)":
                    logs.append(f"⚠️ スキップ: '{file_name}' - 判定点がどの部屋のSTL内にも存在しませんでした。")
                    continue

            except Exception as e:
                logs.append(f"❌ 空間位置判定エラー: {file_name} ({e})")
                continue
        
        elif file_info['type'] == 'vent_in':
            # 給気CSV：[外気 ⇄ 指定部屋] に強制固定
            found_plus_room = file_info['target_room']
            found_minus_room = "外気"
            df[flow_col] = df[flow_col].abs()
            logs.append(f"🔧 換気処理(給気): '{file_name}' を [外気 ⇄ {found_plus_room}] として処理します。")

        elif file_info['type'] == 'vent_out':
            # 排気CSV：[指定部屋 ⇄ 外気] に強制固定
            found_plus_room = "外気"
            found_minus_room = file_info['target_room']
            df[flow_col] = df[flow_col].abs()
            logs.append(f"🔧 換気処理(排気): '{file_name}' を [{found_minus_room} ⇄ 外気] として処理します。")
        # --- (C) 熱量・風量の集計計算 ---
        try:
            # 熱計算
            flow_abs = df[flow_col].abs()
            if flow_abs.sum() > 0:
                mean_temp = (df[temp_col] * flow_abs).sum() / flow_abs.sum()
            else:
                mean_temp = df[temp_col].mean()

            df['heat_kjh'] = df[flow_col] * rho * cp * df[temp_col]
            net_heat_watt = df['heat_kjh'].sum() * 1000 / 3600
            
            # 流量計算
            gross_positive_flow = df[df[flow_col] > 0][flow_col].sum()
            gross_negative_flow = df[df[flow_col] < 0][flow_col].sum()

            opening_results_list.append({
                '開口部': file_key,
                '方向': detected_axis,
                'Plus_Room': found_plus_room,
                'Minus_Room': found_minus_room,
                '平均温度[℃](風量加重平均)': mean_temp,
                '総プラス流量[m3/h]': gross_positive_flow,
                '総マイナス流量[m3/h]': gross_negative_flow,
                '移動熱量[W]': net_heat_watt
            })
            logs.append(f"✅ 計算成功: {file_name} [{found_plus_room} ⇄ {found_minus_room}]")

        except Exception as e:
            logs.append(f"❌ 計算エラー: {file_name} ({e})")

    # 結果をDataFrame化
    if not opening_results_list:
        return None, None, None, logs
    
    results_df = pd.DataFrame(opening_results_list)

    # --- 集計処理 ---
    # 1. 熱収支集計
    heat_movements = []
    df_heat_pos = results_df[results_df['移動熱量[W]'] > 0]
    heat_movements.append(pd.DataFrame({'室名': df_heat_pos['Minus_Room'], '方向': '流出', '熱量[W]': df_heat_pos['移動熱量[W]']}))
    heat_movements.append(pd.DataFrame({'室名': df_heat_pos['Plus_Room'], '方向': '流入', '熱量[W]': df_heat_pos['移動熱量[W]']}))
    
    df_heat_neg = results_df[results_df['移動熱量[W]'] < 0]
    heat_movements.append(pd.DataFrame({'室名': df_heat_neg['Plus_Room'], '方向': '流出', '熱量[W]': df_heat_neg['移動熱量[W]'].abs()}))
    heat_movements.append(pd.DataFrame({'室名': df_heat_neg['Minus_Room'], '方向': '流入', '熱量[W]': df_heat_neg['移動熱量[W]'].abs()}))
    
    # 外部(未特定)を除外して集計
    all_heat_movements = pd.concat(heat_movements)
    all_heat_movements = all_heat_movements[all_heat_movements['室名'] != "外部(未特定)"]
    
    heat_df = all_heat_movements.groupby(['室名', '方向'])['熱量[W]'].sum().unstack(fill_value=0)
    room_heat_summary_df = pd.DataFrame({
        '総流出熱量[W]': heat_df.get('流出', 0),
        '総流入熱量[W]': heat_df.get('流入', 0),
        '処理熱量[W]': heat_df.get('流出', 0) - heat_df.get('流入', 0)
    }).reset_index()

    # 2. 風量収支集計
    flow_movements = []
    flow_movements.append(pd.DataFrame({'室名': results_df['Minus_Room'], '方向': '流出', '流量[m3/h]': results_df['総プラス流量[m3/h]']}))
    flow_movements.append(pd.DataFrame({'室名': results_df['Plus_Room'], '方向': '流入', '流量[m3/h]': results_df['総プラス流量[m3/h]']}))
    flow_movements.append(pd.DataFrame({'室名': results_df['Plus_Room'], '方向': '流出', '流量[m3/h]': results_df['総マイナス流量[m3/h]'].abs()}))
    flow_movements.append(pd.DataFrame({'室名': results_df['Minus_Room'], '方向': '流入', '流量[m3/h]': results_df['総マイナス流量[m3/h]'].abs()}))

    all_flow_movements = pd.concat(flow_movements)
    all_flow_movements = all_flow_movements[all_flow_movements['室名'] != "外部(未特定)"]

    flow_df = all_flow_movements.groupby(['室名', '方向'])['流量[m3/h]'].sum().unstack(fill_value=0)
    room_flow_summary_df = pd.DataFrame({
        '総流出流量[m3/h]': flow_df.get('流出', 0),
        '総流入流量[m3/h]': flow_df.get('流入', 0),
        '風量収支[m3/h]': flow_df.get('流出', 0) - flow_df.get('流入', 0)
    }).reset_index()

    return results_df, room_heat_summary_df, room_flow_summary_df, logs

def create_heat_chart(room_heat_summary_df, fig_width, fig_height, font_size, y_max, custom_colors, show_legend, category_map, mode):
    if "暖房" in mode:
        label_passive = "各室熱損失"
        label_active = "投入熱量"
        passive = room_heat_summary_df[room_heat_summary_df['処理熱量[W]'] < 0].set_index('室名')['処理熱量[W]'].abs()
        active = room_heat_summary_df[room_heat_summary_df['処理熱量[W]'] > 0].set_index('室名')['処理熱量[W]']
    else: 
        label_passive = "各室負荷"
        label_active = "処理熱量"
        passive = room_heat_summary_df[room_heat_summary_df['処理熱量[W]'] > 0].set_index('室名')['処理熱量[W]']
        active = room_heat_summary_df[room_heat_summary_df['処理熱量[W]'] < 0].set_index('室名')['処理熱量[W]'].abs()
        
    plot_df_base = pd.DataFrame({label_passive: passive , label_active: active}).T.fillna(0)

    desired_order = []
    for rooms in category_map.values():
        desired_order.extend(rooms)
    
    current_columns = plot_df_base.columns.tolist()
    ordered_columns = [col for col in desired_order if col in current_columns]
    remaining_columns = [col for col in current_columns if col not in desired_order]
    final_column_order = ordered_columns + remaining_columns
    
    plot_df = plot_df_base[final_column_order]
    
    colors = []
    default_color = '#AAAAAA'
    for room in final_column_order:
        colors.append(custom_colors.get(room, default_color))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plot_df.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.8, legend=False)

    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', alpha=0.7, color='#cccccc')
    ax.grid(axis='x', visible=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax.tick_params(axis='y', length=0, labelsize=font_size)
    ax.tick_params(axis='x', length=0)
    plt.xticks(rotation=0, fontsize=font_size)
    plt.ylabel('処理熱量[W]', fontsize=font_size)
    
    if y_max > 0:
        ax.set_ylim(0, y_max)

    plt.axhline(0, color='black', linewidth=0.8)

    for i, container in enumerate(ax.containers):
        labels = [f"{v:,.0f}" if v > 0 else '' for v in container.datavalues]
        ax.bar_label(container, labels=labels, label_type='center', color='black', fontsize=font_size*0.8, fontweight='bold')

    if show_legend:
        handles, labels_legend = ax.get_legend_handles_labels()
        new_handles = []
        new_labels = []
        dummy_handle = mpatches.Patch(visible=False)

        for category_name, rooms_in_category in reversed(category_map.items()):
            category_handles_labels = []
            for room_name in reversed(rooms_in_category):
                if room_name in labels_legend:
                    index = labels_legend.index(room_name)
                    category_handles_labels.append((handles[index], f"  {room_name}"))
            
            if category_handles_labels:
                if category_name:
                    new_handles.append(dummy_handle)
                    new_labels.append(f"--- {category_name} ---")
                for handle, label in category_handles_labels:
                    new_handles.append(handle)
                    new_labels.append(label)

        remaining_items = [(handles[i], f"  {labels_legend[i]}") for i, label in enumerate(labels_legend) if label not in desired_order]
        if remaining_items:
            new_handles.append(dummy_handle)
            new_labels.append("▼ 未分類")
            for handle, label in remaining_items:
                new_handles.append(handle)
                new_labels.append(label)

        ax.legend(handles=new_handles, labels=new_labels, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=font_size*0.9)

    total_pos = passive.sum()
    total_neg = active.sum()
    
    return fig, total_pos, total_neg

# ==========================================
# 2. アプリケーション UI
# ==========================================

st.set_page_config(page_title="CFD搬送熱量分析ツール (流出入自動判定版)", layout="wide")

st.title("CFD 搬送熱量 & 風量バランス分析 (流出入自動判定版)")
st.markdown("部屋のSTLデータを用いて、開口部CSVが面する部屋を特定します。")

if 'uploader_key' not in st.session_state:
    st.session_state['uploader_key'] = 0
        
with st.sidebar:
    st.header("1. 解析設定")
    mode = st.radio("モード", ["冷房", "暖房"])
    st.divider()
    
    st.header("2. 定数設定")
    rho = st.number_input("空気密度 ρ [kg/m3]", value=1.20)
    cp = st.number_input("比熱 Cp [J/g・K]", value=1.006, format="%.3f")
    threshold = st.number_input("風量収支許容誤差 [m3/h]", value=1.0, format="%.2f", help="風量収支チェックに用いる値で、部屋に流出入する風量の差がこれより大きくなったとき警告を表示します。")
    # 判定点をずらす距離を調整可能に
    offset_dist = st.number_input("STL判定のオフセット距離 [m]", value=0.05, step=0.01, format="%.2f", help="開口部が面する2部屋の認識に用いています。開口部の中心から表裏2方向の法線方向にこの距離だけ移動した点が、どの部屋の領域に含まれているかを判定します。")
    st.divider()
    
    st.header("3. 分析ファイル")
    st.info("部屋ボリュームのSTLファイルをすべてアップロード")
    stl_files = st.file_uploader("部屋のSTLデータ (複数選択)", type="stl", accept_multiple_files=True)
    st.markdown("---")

    st.info("FlowDesignerで書き出した開口部のCSVをすべてアップロード")
    cfd_files = st.file_uploader(
        "CFD解析結果 (複数選択)",
        type="csv",
        accept_multiple_files=True,
        key = f"cfd_uploader_{st.session_state['uploader_key']}"
    )

    def reset_files():
        st.session_state['uploader_key'] += 1
        st.session_state['analyzed'] = False

    if st.button("リセット"):
        reset_files()
        st.rerun()
    st.divider()
    
    st.header("4. 換気個別設定 (オプション)")
    st.markdown("給排気口が直接外気（外壁）に接していない場合、給気・排気のCSVを個別に指定します。")
    st.info("⚠️ここに入力するCSVファイルは「3.分析ファイル」のCFD解析結果には入れない")
    
    st.subheader("給気（外気 → 室内）")
    in_file = st.file_uploader("給気口CSVファイルをアップロード", type="csv", key="vent_in_uploader")
    in_room = st.text_input("SAが流入する部屋名（STL名と一致させてください）", value="床下")
    
    st.subheader("垂直排気（室内 → 外気）")
    out_file = st.file_uploader("排気口CSVファイルをアップロード", type="csv", key="vent_out_uploader")
    out_room = st.text_input("EAが流出する部屋名（STL名と一致させてください）", value="ホール")
# --- メイン処理 ---

if 'analyzed' not in st.session_state:
    st.session_state['analyzed'] = False
    st.session_state['results_df'] = None
    st.session_state['room_heat_df'] = None
    st.session_state['room_flow_df'] = None
    st.session_state['logs'] = []

if st.button("解析実行", type="primary"):
    if not cfd_files and not in_file and not out_file:
        st.warning("CFD解析結果のCSVをアップロードしてください。")
    else:
        with st.spinner("計算中..."):
            # 1. 換気用の設定を辞書にまとめる
            vent_settings = {
                'in_file': in_file,
                'in_room': in_room,
                'out_file': out_file,
                'out_room': out_room
            }
            
            # 2. 関数の引数の最後に「vent_settings」を追加して呼び出す
            # (※ stl_files はローカル自動読み込みにしたため不要になっています)
            results_df, room_heat_df, room_flow_df, logs = process_cfd_files_with_stl(
                stl_files, cfd_files, rho, cp, threshold, offset_dist, vent_settings
            )
            
            st.session_state['logs'] = logs
            if results_df is not None:
                st.session_state['results_df'] = results_df
                st.session_state['room_heat_df'] = room_heat_df
                st.session_state['room_flow_df'] = room_flow_df
                st.session_state['analyzed'] = True
                st.success("解析完了")
            else:
                st.error("有効なデータが作成されませんでした。ログを確認してください。")

# ログ表示コンテナ（常時確認できるようにボタンの下に配置）
if st.session_state['logs']:
    with st.expander("実行ログ・エラー・警告", expanded=not st.session_state['analyzed']):
        for log in st.session_state['logs']:
            if "❌" in log: st.error(log)
            elif "⚠️" in log: st.warning(log)
            elif "✅" in log: st.success(log)
            else: st.info(log)

if st.session_state['analyzed']:
    results_df = st.session_state['results_df']
    room_heat_df = st.session_state['room_heat_df']
    room_flow_df = st.session_state['room_flow_df']

    tab1, tab2, tab3 = st.tabs(["風量収支チェック", "熱量分配グラフ", "計算詳細"])

    # --- Tab 1: 風量バランス ---
    with tab1:
        st.subheader("風量収支チェック")
        st.caption(f"許容誤差: ±{threshold} m3/h")
        
        warning_count = 0
        for index, row in room_flow_df.iterrows():
            room = row['室名']
            balance = row['風量収支[m3/h]']
            
            if balance > threshold:
                st.error(f"⚠️ {room}: 流出過多 (流入不足) +{balance:.2f} m3/h")
                warning_count += 1
            elif balance < -threshold:
                st.error(f"⚠️ {room}: 流入過多 (流出不足) {balance:.2f} m3/h")
                warning_count += 1
            else:
                st.success(f"{room}: OK ({balance:+.2f} m3/h)")
        
        if warning_count == 0:
            st.info("✅ 全室で風量収支が許容値以下")

    # --- Tab 2: グラフ ---
    with tab2:
        st.subheader("各室およびエアコンの空調処理熱量")
        all_rooms = sorted(room_heat_df['室名'].unique())

        with st.expander("グラフをカスタマイズする", expanded=False):
            st.markdown("#### 凡例グループと並び順")
            default_categories_list = [
                ("１階", ["SCL", "玄関", "トイレ", "洗面室", "階段", "LDK"]),
                ("２階", ["洋室2", "洋室1", "WCL", "主寝室", "吹抜", "ホール"]),
                ("その他", ["床下", "小屋裏", "階間"]),
                ("空調機", ["AC1F", "AC2F"])
            ]

            num_categories = st.number_input("カテゴリー数", min_value = 1, max_value = 10, value = 4, step = 1)
            custom_category_map = {}
            cols_cat = st.columns(4)

            for i in range(num_categories):
                with cols_cat[i % 4]:
                    if i < len(default_categories_list):
                        def_name = default_categories_list[i][0]
                        def_rooms = [r for r in default_categories_list[i][1] if r in all_rooms]
                    else:
                        def_name  = f"グループ{i+1}"
                        def_rooms = []

                    cat_name = st.text_input(f"カテゴリ名{i+1}", value=def_name, key=f"cat_name_{i}")
                    selected_rooms = st.multiselect(f"{cat_name} の部屋", options=all_rooms, default=def_rooms, key=f"cat_rooms_{i}")
                    if cat_name and selected_rooms:
                        custom_category_map[cat_name] = selected_rooms

            st.divider()
            st.markdown("#### グラフ体裁")
            col_ui1, col_ui2, col_ui3 = st.columns(3)
            
            with col_ui1:
                st.markdown("**サイズ設定**")
                fig_w = st.number_input("横幅 (inch)", value=6.0, step=0.5)
                fig_h = st.number_input("高さ (inch)", value=10.0, step=0.5)
            
            with col_ui2:
                st.markdown("**表示設定**")
                font_size = st.slider("文字サイズ", 8, 40, 14)
                y_max = st.number_input("Y軸の最大値 (0で自動)", value=0, step=100)
                show_legend = st.checkbox("凡例を表示する", value=True)

            with col_ui3:
                st.markdown("**色の設定**")
                default_colors = {
                    "LDK": "#FF6F6F", "1階": "#FF7F50", "2階": "#0000FF", "廊下": "#9370DB",
                    "洋室1": "#6495ED", "洋室2": "#FFA500", "R3": "#32CD32", "床下": "#D3D3D3",
                    "AC": "#87CEEB", "AC1F": "#87CEEB","AC2F": "#1D8EFF", "洗面室": "#40E0D0",
                    "和室": "#BDB76B", "SR": "#FFFF00","LD": "#7DA055", "小屋裏": "#FFD330",
                    "2階廊下等": "#7DC6BE", "キッチン": "#E2E878","主寝室": "#9BCE8A",
                    "階段": "#F39C60", "ホール": "#7DC6BE","吹抜": "#D16500",
                    "玄関": "#82B000", "階間": "#5E5E5E",
                    "SCL": "#425102", "WCL": "#6E833C",
                }
                custom_colors = {}
                if st.checkbox("色を個別に変更する"):
                    for room in all_rooms:
                        initial = default_colors.get(room, "#AAAAAA")
                        custom_colors[room] = st.color_picker(f"{room}", value=initial, key=f"color_{room}")
                else:
                    custom_colors = default_colors

        try:
            fig, total_passive, total_active = create_heat_chart(
                room_heat_df, fig_w, fig_h, font_size, y_max, custom_colors, show_legend, custom_category_map, mode
            )
            st.pyplot(fig)
            
            col1, col2 = st.columns(2)
            label_left = "各室熱損失合計" if "暖房" in mode else "各室熱負荷合計"
            label_right = "投入熱量" if "暖房" in mode else "処理熱量"
            col1.metric(label_left, f"{total_passive:,.1f} W")
            col2.metric(label_right, f"{total_active:,.1f} W")
            
            img = io.BytesIO()
            fig.savefig(img, format='svg', bbox_inches='tight')
            st.download_button("グラフをSVGで保存", img, "heat_balance.svg", "image/svg+xml")
            
        except Exception as e:
            st.error(f"グラフ作成エラー: {e}")

    # --- Tab 3: 計算詳細 ---
    with tab3:
        st.markdown("### 📥 データダウンロード")
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        col_dl1.download_button("表1 (開口部風量・移動熱量)", results_df.to_csv(index=False).encode('utf-8-sig'), "results_raw.csv")
        col_dl2.download_button("表2 (処理熱量)", room_heat_df.to_csv(index=False).encode('utf-8-sig'), "results_heat.csv")
        col_dl3.download_button("表3 (風量収支)", room_flow_df.to_csv(index=False).encode('utf-8-sig'), "results_flow.csv")
        st.divider()

        st.markdown("### (表1) 開口部別 風量・移動熱量")
        st.dataframe(results_df)
        
        st.markdown("### (表2) 室別 処理熱量")
        st.dataframe(room_heat_df)
        
        st.markdown("### (表3) 室別 風量収支")
        st.dataframe(room_flow_df)
