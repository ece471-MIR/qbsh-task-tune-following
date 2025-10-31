"""
Data Loader for MIR-QBSH Corpus
"""

import os 
import numpy as np 
from pathlib import Path 
from typing import Dict, List, Tuple 
import mido 

class MIRQBSHDataset:

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.midi_dir = self.data_root / "midiFile"
        self.wave_dir = self.data_root / "waveFile"

        self.song_list_path = self.midi_dir / "songList.txt"
        self.song_list = self._load_song_list()
        self.query_files = self._find_all_queries()

        print(f"Loaded dataset:")
        print(f"  - {len(self.song_list)} ground-truth songs")
        print(f"  - {len(self.query_files)} query files")

    
    def _load_song_list(self) -> Dict[str, Dict]:
        songs = {}

        with open(self.song_list_path, 'r', encoding='big5') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    filename = parts[0]
                    song_key = filename.replace('.mid', '')

                    songs[song_key] = {
                        'filename': filename,
                        'english_title': parts[1] if parts [1] != '-' else None,
                        'chinese_title': parts[2] if parts[2] != '-' else None,
                        'num_recordings': int(parts[3])
                     }

        return songs    


    def _find_all_queries(self) -> List[Path]:
        query_files = []
        for year_dir in self.wave_dir.iterdir():
            if not year_dir.is_dir():
                continue

            for person_dir in year_dir.iterdir():
                if not person_dir.is_dir():
                    continue

                pv_files = list(person_dir.glob("*.pv"))
                query_files.extend(pv_files)

        return sorted(query_files)

    def load_query_pv(self, filepath: Path) -> np.ndarray:
        pitches = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        pitch = float(line)
                        pitches.append(pitch)
                    except ValueError:
                        continue

        return np.array(pitches, dtype=np.float64)


    def load_template_midi(self, song_key: str, frame_rate: float = 31.25) -> np.ndarray:
        if song_key in self.song_list:
            midi_filename = self.song_list[song_key]['filename']
            if not midi_filename.endswith('.mid'):
                midi_filename = midi_filename + '.mid'
        else:
            midi_filename = song_key + '.mid'

        midi_path = self.midi_dir / midi_filename
        
        midi = mido.MidiFile(midi_path)

        pitch_vector = self._midi_to_pitch_vector(midi, frame_rate)
        
        return pitch_vector

    def _midi_to_pitch_vector(self, midi: mido.MidiFile, frame_rate: float) -> np.ndarray:   
        ticks_per_beat = midi.ticks_per_beat
        tempo = 500000

        for track in midi.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                    break

        seconds_per_tick = (tempo / 1000000) / ticks_per_beat

        notes = []
        current_time = 0

        for track in midi.tracks:
            track_time = 0
            for msg in track:
                track_time += msg.time

                if msg.type == 'note_on' and msg.velocity > 0:
                    time_seconds = track_time * seconds_per_tick
                    notes.append({
                        'time': time_seconds,
                        'pitch': msg.note,
                        'type': 'on'
                    })     
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    time_seconds = track_time * seconds_per_tick
                    notes.append({
                        'time': time_seconds,
                        'pitch': msg.note,
                        'type': 'off'
                    })

        if not notes:
            return np.array([])
        
        notes.sort(key=lambda x: x['time'])
        
        max_time = notes[-1]['time']
        num_frames = int(max_time * frame_rate) + 1
        
        pitch_vector = np.zeros(num_frames)
        active_notes = set()
        
        note_idx = 0
        for frame_idx in range(num_frames):
            frame_time = frame_idx / frame_rate
            
            while note_idx < len(notes) and notes[note_idx]['time'] <= frame_time:
                note = notes[note_idx]
                if note['type'] == 'on':
                    active_notes.add(note['pitch'])
                else:
                    active_notes.discard(note['pitch'])
                note_idx += 1
            
            if active_notes:
                pitch_vector[frame_idx] = max(active_notes)
            else:
                pitch_vector[frame_idx] = 0
        
        return pitch_vector
    
    def get_ground_truth_mapping(self) -> Dict[str, str]:
        ground_truth = {}
        
        for query_path in self.query_files:

            query_name = query_path.stem  

            if query_name in self.song_list:
                ground_truth[str(query_path)] = query_name
            else:
                midi_name = query_name + '.mid'
                for song_key, song_info in self.song_list.items():
                    if song_info['filename'] == midi_name:
                        ground_truth[str(query_path)] = song_key
                        break
        
        return ground_truth
    
    def load_all_templates(self, frame_rate: float = 31.25) -> Dict[str, np.ndarray]:
        templates = {}
        
        print("Loading templates...")
        for song_key in self.song_list:
            try:
                pitch_vector = self.load_template_midi(song_key, frame_rate)
                templates[song_key] = pitch_vector
                print(f"  Loaded: {song_key} ({len(pitch_vector)} frames)")
            except Exception as e:
                print(f"  Error loading {song_key}: {e}")
        
        return templates            
