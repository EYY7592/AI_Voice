import json
import os

def update_function_in_notebook(file_path):
    print(f"Updating function in {file_path}...")
    if not os.path.exists(file_path):
        print(f"Error: {file_path} does not exist.")
        return
        
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    new_func_code = [
        "def extract_prosody_features(audio, sr=16000):\n",
        "    \"\"\"從音頻萃取 20 維韻律特徵向量 (優化穩定版)\n",
        "    \n",
        "    支援 Numpy/Torch Tensor (CPU/GPU) 以及單/多聲道輸入。\n",
        "    \"\"\"\n",
        "    try:\n",
        "        # 1. 前處理：轉換為 Numpy 單聲道\n",
        "        if hasattr(audio, 'detach'): # Torch Tensor\n",
        "            audio = audio.detach().cpu().numpy()\n",
        "        \n",
        "        audio = np.atleast_1d(audio)\n",
        "        if audio.ndim > 1:\n",
        "            if audio.shape[0] < audio.shape[1]:\n",
        "                audio = np.mean(audio, axis=0)\n",
        "            else:\n",
        "                audio = np.mean(audio, axis=1)\n",
        "        \n",
        "        # 2. 建立 Praat Sound 物件\n",
        "        snd = parselmouth.Sound(audio, sampling_frequency=sr)\n",
        "        \n",
        "        # 3. Pitch 分析 (F0)\n",
        "        pitch = call(snd, 'To Pitch', 0.0, 75, 600)\n",
        "        f0_values = pitch.selected_array['frequency']\n",
        "        f0_voiced = f0_values[f0_values > 0]\n",
        "        \n",
        "        f0_mean = float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else 0.0\n",
        "        f0_std = float(np.std(f0_voiced)) if len(f0_voiced) > 0 else 0.0\n",
        "        f0_range = float(np.ptp(f0_voiced)) if len(f0_voiced) > 0 else 0.0\n",
        "        \n",
        "        # 4. Jitter / Shimmer / HNR\n",
        "        point_process = call(snd, 'To PointProcess (periodic, cc)', 75, 600)\n",
        "        jitter = call(point_process, 'Get jitter (local)', 0, 0, 0.0001, 0.02, 1.3)\n",
        "        shimmer = call([snd, point_process], 'Get shimmer (local)', 0, 0, 0.0001, 0.02, 1.3, 1.6)\n",
        "        \n",
        "        harmonicity = call(snd, 'To Harmonicity (cc)', 0.01, 75, 0.1, 1.0)\n",
        "        hnr = call(harmonicity, 'Get mean', 0, 0)\n",
        "        \n",
        "        # 5. Formant (F1-F4)\n",
        "        formant = call(snd, 'To Formant (burg)', 0.0, 5, 5500, 0.025, 50)\n",
        "        formants = []\n",
        "        for i in range(1, 5):\n",
        "            try:\n",
        "                f = call(formant, 'Get mean', i, 0, 0, 'hertz')\n",
        "                formants.append(f if not np.isnan(f) else 0.0)\n",
        "            except:\n",
        "                formants.append(0.0)\n",
        "        \n",
        "        # 6. Speaking rate & Pause (基於能量)\n",
        "        rms = librosa.feature.rms(y=audio, frame_length=512, hop_length=160)[0]\n",
        "        threshold = np.mean(rms) * 0.5\n",
        "        voiced_frames = np.sum(rms > threshold)\n",
        "        total_frames = len(rms)\n",
        "        speaking_rate = voiced_frames / max(1, total_frames) * (sr / 160)\n",
        "        \n",
        "        silent_mask = rms <= threshold\n",
        "        pause_durations = []\n",
        "        count = 0\n",
        "        for is_silent in silent_mask:\n",
        "            if is_silent:\n",
        "                count += 1\n",
        "            elif count > 0:\n",
        "                pause_durations.append(count * 160 / sr)\n",
        "                count = 0\n",
        "        \n",
        "        pause_mean = float(np.mean(pause_durations)) if pause_durations else 0.0\n",
        "        pause_std = float(np.std(pause_durations)) if pause_durations else 0.0\n",
        "        pause_count = len(pause_durations)\n",
        "        f0_cv = f0_std / f0_mean if f0_mean > 0 else 0.0\n",
        "        \n",
        "        # 7. 建立 20 維特徵向量\n",
        "        raw_vector = [\n",
        "            jitter, shimmer, hnr,\n",
        "            f0_mean, f0_std, f0_range,\n",
        "            speaking_rate,\n",
        "            pause_mean, pause_std, pause_count,\n",
        "            *formants, f0_cv,\n",
        "            jitter * shimmer,\n",
        "            hnr / max(f0_mean, 1),\n",
        "            f0_range / max(f0_std, 0.01),\n",
        "            pause_count / max(len(audio) / sr, 0.1),\n",
        "            float(np.mean(np.abs(np.diff(f0_voiced)))) if len(f0_voiced) > 1 else 0.0,\n",
        "        ]\n",
        "        \n",
        "        # 8. 清理 NaN / Inf\n",
        "        return np.nan_to_num(np.array(raw_vector, dtype=np.float32), nan=0.0)\n",
        "    \n",
        "    except Exception as e:\n",
        "        print(f'特徵萃取失敗: {e}')\n",
        "        return np.zeros(20, dtype=np.float32)\n"
    ]

    changed = False
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            # Search for the function definition
            has_func = any('def extract_prosody_features' in line for line in source)
            if has_func:
                cell['source'] = new_func_code
                changed = True
                print(" Found and replaced function.")

    if changed:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("Success.")
    else:
        print("Function not found in notebook.")

# Target
target = r"d:\code\side-project_VC\AI_Voice\notebooks\02_voiceprint_training.ipynb"
update_function_in_notebook(target)
