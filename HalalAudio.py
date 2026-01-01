import os, random, sys, math, argparse
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm

work = os.path.dirname(os.path.abspath(__file__))
sounds = os.path.join(work, "sounds")

# хуйня для выбора звука по рандому
def randsound(max_len=8000, gain=16):
    if not os.path.exists(sounds): return None
    files = [f for f in os.listdir(sounds) if f.lower().endswith(('.mp3', '.wav', '.ogg'))]
    if not files: return None
    try:
        m = AudioSegment.from_file(os.path.join(sounds, random.choice(files))).set_channels(2)
        return m[:max_len] + gain
    except: return None

# всякие эффекты
def ef1(segment):
    chunks = [segment[i:i+50] for i in range(0, len(segment), 50)]
    processed = []
    for i, chunk in enumerate(chunks):
        pan = math.sin(i)
        processed.append(chunk.pan(pan))
    return sum(processed)

def ef2(segment):
    samples = np.array(segment.get_array_of_samples())
    crushed = (samples // 4000) * 4000
    return segment._spawn(crushed.astype(samples.dtype).tobytes())

def ef3(segment):
    octaves = random.choice([-0.5, 0.5, 1.0, -0.8])
    new_sample_rate = int(segment.frame_rate * (2.0 ** octaves))
    return segment._spawn(segment.raw_data, overrides={'frame_rate': new_sample_rate}).set_frame_rate(44100)

def bass(segment, strength):
    lows = segment.low_pass_filter(180)
    return segment.overlay(lows + strength).overlay(lows + (strength - 5))

def rand_ef(segment, ear_rape_vol):
    choice = random.random()
    if choice < 0.10: return segment.reverse()
    if choice < 0.20: return ef1(segment)
    if choice < 0.30: return ef2(segment)
    if choice < 0.40: return ef3(segment)
    if choice < 0.50: return segment.overlay(segment - 5, position=random.randint(50, 500))
    if choice < 0.60: return segment.high_pass_filter(random.randint(1000, 3000))
    if choice < 0.70: 
        chunk = segment[:100]
        return (chunk * 20)[:len(segment)]
    if choice < 0.85: return segment + ear_rape_vol 
    return segment

def aistart(input_path, output_path, bass_strength, freq):
    try:
        audio = AudioSegment.from_file(input_path).set_channels(2)
    except Exception as e:
        print(f"❌ oшибка: {e}"); return

    pbar = tqdm(total=100, desc="HalalAudio AI обрабатывает файл")

    num_ints = int(random.randint(10, 20) * freq * 2)
    for _ in range(num_ints):
        snd = randsound()
        if snd:
            pos = random.randint(500, max(501, len(audio)-500))
            audio = audio[:pos] + snd + audio[pos:]
    pbar.update(20)

    num_overlays = int(random.randint(20, 45) * freq * 2)
    for _ in range(num_overlays):
        snd = randsound(5000)
        if snd:
            rate = int(snd.frame_rate * random.uniform(0.4, 2.5))
            snd = snd._spawn(snd.raw_data, overrides={'frame_rate': rate}).set_frame_rate(44100)
            audio = audio.overlay(snd, position=random.randint(0, len(audio)))
    pbar.update(20)

    chunk_len = 1000 
    processed = AudioSegment.empty()
    for i in range(0, len(audio), chunk_len):
        chunk = audio[i:i+chunk_len]
        if random.random() < freq: 
            chunk = rand_ef(chunk, bass_strength // 2)
        if random.random() < (freq * 0.8): 
            chunk = bass(chunk, bass_strength)
        processed += chunk
    audio = processed
    pbar.update(30)

    if random.random() < (0.15 * freq):
        samples = np.array(audio.get_array_of_samples())
        x = np.linspace(0, 5 * np.pi * (len(audio)/1000), len(samples))
        lfo = (np.sin(x) + 1.1) / 2.1
        audio = audio._spawn((samples * lfo).astype(samples.dtype))
    pbar.update(10)

    audio = audio + (bass_strength // 3)
    audio = audio.set_frame_rate(random.choice([8000, 11025, 22050])).set_frame_rate(44100)

    audio.export(output_path, format="mp3", bitrate="128k")
    pbar.update(20)
    pbar.close()
    print(f"\n✅ готово")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HalalAudio AI v1")
    parser.add_argument("input", help="Путь к исходному файлу")
    parser.add_argument("-b", "--bass", type=int, default=22, help="вручную выбрать силу баса")
    parser.add_argument("-a", "--freq", type=float, default=0.5, help="вручную выбрать частоту появления эффектов (0.1 - 1.0)")

    args = parser.parse_args()

    inp = args.input
    out = f"halalAUDIO_{os.path.basename(inp)}"
    aistart(inp, out, args.bass, args.freq)


