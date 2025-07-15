import os
import tempfile
import torch
import whisper
import librosa
import soundfile as sf
import numpy as np
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from pyannote.audio import Pipeline
import tempfile
import os
from huggingface_hub import login

class TranscriptionView(APIView):
    def __init__(self):
        super().__init__()
        # Initialize models as None - they'll be loaded lazily
        self.whisper_model = None
        self.setup_ffmpeg_path()
        
    def setup_ffmpeg_path(self):
        """Add local ffmpeg to PATH for this process"""
        import sys
        from pathlib import Path
        
        # Get the project root directory
        project_root = Path("C:/ffmpeg")
        ffmpeg_bin_path = project_root
        
        # Add ffmpeg to PATH if it exists
        if ffmpeg_bin_path.exists():
            ffmpeg_bin_str = str(ffmpeg_bin_path)
            current_path = os.environ.get('PATH', '')
            
            # Only add if not already in PATH
            if ffmpeg_bin_str not in current_path:
                os.environ['PATH'] = ffmpeg_bin_str + os.pathsep + current_path
                print(f"✅ Added local ffmpeg to PATH: {ffmpeg_bin_str}")
            
            # Verify ffmpeg is working
            self.verify_ffmpeg()
        else:
            print(f"❌ ffmpeg directory not found at: {ffmpeg_bin_path}")
            print("Please create the ffmpeg/bin directory and place ffmpeg.exe there")
    
    def verify_ffmpeg(self):
        """Verify that ffmpeg is available"""
        import subprocess
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ ffmpeg is working correctly")
                return True
            else:
                print("❌ ffmpeg found but returned error")
                return False
        except FileNotFoundError:
            print("❌ ffmpeg not found in PATH")
            return False
        except subprocess.TimeoutExpired:
            print("❌ ffmpeg command timed out")
            return False
        except Exception as e:
            print(f"❌ ffmpeg verification failed: {e}")
            return False
        
    def load_whisper_model(self):
        """Load Whisper model optimized for CPU usage"""
        if self.whisper_model is None:
            print("Loading Whisper model (CPU optimized)...")
            # Use 'tiny' or 'base' model for CPU - good balance of speed and accuracy
            # tiny: fastest, base: good quality, small: better quality but slower
            model_size = "base"  # Change to "tiny" for faster processing
            
            # Force CPU usage
            device = "cpu"
            torch.set_num_threads(4)  # Optimize for CPU
            
            self.whisper_model = whisper.load_model(model_size, device=device)
            print(f"Loaded Whisper {model_size} model on CPU")

    def advanced_speaker_diarization(self, audio_data, segments, sample_rate=16000):
        """
        Improved speaker diarization that handles pauses correctly and groups segments by speaker
        """
        try:
            # First, extract features for all segments
            segment_features = []
            valid_segments = []
            
            for segment in segments:
                start_time = segment['start']
                end_time = segment['end']
                
                # Extract audio segment
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                segment_audio = audio_data[start_sample:end_sample]
                
                if len(segment_audio) > 1600:  # At least 0.1 seconds
                    features = self.extract_speaker_features(segment_audio, sample_rate)
                    if features is not None:
                        segment_features.append(features)
                        valid_segments.append(segment)
            
            if not segment_features:
                # No valid segments, return single speaker
                return [{
                    'speaker': 'Speaker_1',
                    'text': segment['text'].strip(),
                    'start': int(segment['start'] * 1000),
                    'end': int(segment['end'] * 1000),
                    'confidence': 0.9
                } for segment in segments]
            
            # Determine number of speakers using clustering
            speaker_labels = self.cluster_speakers(segment_features)
            unique_speakers = len(set(speaker_labels))
            
            print(f"Detected {unique_speakers} speaker(s) from {len(valid_segments)} segments")
            
            # Create initial speaker assignments
            initial_speakers = []
            for i, (segment, speaker_label) in enumerate(zip(valid_segments, speaker_labels)):
                initial_speakers.append({
                    'speaker': f'Speaker_{speaker_label + 1}',
                    'text': segment['text'].strip(),
                    'start': int(segment['start'] * 1000),
                    'end': int(segment['end'] * 1000),
                    'confidence': 0.8 if unique_speakers == 1 else 0.7,
                    'original_segment': segment
                })
            
            # Now merge consecutive segments from the same speaker
            merged_speakers = self.merge_consecutive_speakers(initial_speakers)
            
            # Handle any segments that were filtered out during feature extraction
            all_processed_segments = {seg['original_segment']['start'] for seg in initial_speakers}
            unprocessed_segments = [seg for seg in segments if seg['start'] not in all_processed_segments]
            
            if unprocessed_segments:
                most_common_speaker = self.get_most_common_speaker(merged_speakers)
                for segment in unprocessed_segments:
                    merged_speakers.append({
                        'speaker': most_common_speaker,
                        'text': segment['text'].strip(),
                        'start': int(segment['start'] * 1000),
                        'end': int(segment['end'] * 1000),
                        'confidence': 0.6
                    })
            
            # Sort by start time and clean up
            merged_speakers.sort(key=lambda x: x['start'])
            
            # Remove the original_segment field from final output
            for speaker in merged_speakers:
                speaker.pop('original_segment', None)
            
            return merged_speakers
            
        except Exception as e:
            print(f"Advanced diarization failed: {e}")
            # Fallback to single speaker
            return [{
                'speaker': 'Speaker_1',
                'text': segment['text'].strip(),
                'start': int(segment['start'] * 1000),
                'end': int(segment['end'] * 1000),
                'confidence': 0.5
            } for segment in segments]

    def extract_speaker_features(self, segment_audio, sample_rate):
        """Extract comprehensive features for speaker identification"""
        try:
            if len(segment_audio) < 1600:  # Less than 0.1 seconds
                return None
            
            # Normalize audio
            segment_audio = segment_audio / (np.max(np.abs(segment_audio)) + 1e-8)
            
            # Extract multiple features
            features = []
            
            # 1. Spectral centroid (brightness)
            spectral_centroid = librosa.feature.spectral_centroid(y=segment_audio, sr=sample_rate)[0]
            features.append(np.mean(spectral_centroid))
            features.append(np.std(spectral_centroid))
            
            # 2. Zero crossing rate (voice characteristics)
            zcr = librosa.feature.zero_crossing_rate(segment_audio)[0]
            features.append(np.mean(zcr))
            
            # 3. RMS energy
            rms = librosa.feature.rms(y=segment_audio)[0]
            features.append(np.mean(rms))
            
            # 4. Spectral rolloff (voice timbre)
            rolloff = librosa.feature.spectral_rolloff(y=segment_audio, sr=sample_rate)[0]
            features.append(np.mean(rolloff))
            
            # 5. MFCC features (first 5 coefficients)
            mfccs = librosa.feature.mfcc(y=segment_audio, sr=sample_rate, n_mfcc=5)
            for i in range(5):
                features.append(np.mean(mfccs[i]))
            
            # 6. Pitch features
            pitches, magnitudes = librosa.piptrack(y=segment_audio, sr=sample_rate)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                features.append(np.mean(pitch_values))
                features.append(np.std(pitch_values))
            else:
                features.extend([0, 0])
            
            return np.array(features)
            
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            return None

    def cluster_speakers(self, features_list):
        """Cluster segments to identify unique speakers with improved sensitivity"""
        import numpy as np
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        
        try:
            # Convert to numpy array
            features_array = np.array(features_list)
            
            # Normalize features
            scaler = StandardScaler()
            features_normalized = scaler.fit_transform(features_array)
            
            # Calculate pairwise distances to understand data distribution
            from sklearn.metrics.pairwise import euclidean_distances
            distances = euclidean_distances(features_normalized)
            
            # Get statistics about the distances
            upper_triangle = distances[np.triu_indices_from(distances, k=1)]
            mean_distance = np.mean(upper_triangle)
            std_distance = np.std(upper_triangle)
            max_distance = np.max(upper_triangle)
            
            print(f"Distance stats: mean={mean_distance:.3f}, std={std_distance:.3f}, max={max_distance:.3f}")
            
            # More conservative thresholds for single speaker detection
            single_speaker_threshold = 1.8  # Increased threshold
            
            if max_distance < single_speaker_threshold:
                print(f"Single speaker detected (max_distance={max_distance:.3f} < {single_speaker_threshold})")
                return [0] * len(features_list)
            
            # Use DBSCAN with adaptive parameters based on data distribution
            # More relaxed eps to avoid over-segmentation
            eps = max(0.8, mean_distance + 0.5 * std_distance)
            min_samples = max(1, len(features_list) // 4)  # At least 25% of segments to form a cluster
            
            print(f"DBSCAN parameters: eps={eps:.3f}, min_samples={min_samples}")
            
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features_normalized)
            labels = clustering.labels_
            
            # Check results
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = list(labels).count(-1)
            
            print(f"DBSCAN found {n_clusters} clusters, {n_noise} noise points")
            
            if n_clusters <= 1 or n_noise > len(features_list) * 0.5:  # Too much noise
                # Try K-means with 2 clusters as fallback
                from sklearn.cluster import KMeans
                
                # Only try 2 clusters if we have enough data points
                if len(features_list) >= 4:
                    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(features_normalized)
                    
                    # Check if the clusters are significantly different
                    cluster_0_features = features_normalized[labels == 0]
                    cluster_1_features = features_normalized[labels == 1]
                    
                    if len(cluster_0_features) > 0 and len(cluster_1_features) > 0:
                        center_0 = np.mean(cluster_0_features, axis=0)
                        center_1 = np.mean(cluster_1_features, axis=0)
                        center_distance = np.linalg.norm(center_0 - center_1)
                        
                        # More strict threshold for K-means
                        if center_distance < 1.5:  # Clusters too similar
                            print(f"K-means clusters too similar (distance={center_distance:.3f}), single speaker")
                            return [0] * len(features_list)
                        else:
                            # Evaluate cluster quality with silhouette score
                            try:
                                from sklearn.metrics import silhouette_score
                                silhouette = silhouette_score(features_normalized, labels)
                                print(f"K-means silhouette score: {silhouette:.3f}")
                                if silhouette < 0.2:
                                    print("Silhouette score too low, treating as single speaker")
                                    return [0] * len(features_list)
                            except Exception as sil_err:
                                print(f"Silhouette calculation failed: {sil_err}")
                            
                            print(f"K-means detected 2 speakers (distance={center_distance:.3f})")
                            return labels
                    else:
                        return [0] * len(features_list)
                else:
                    print("Not enough segments for reliable clustering, assuming single speaker")
                    return [0] * len(features_list)
            else:
                # DBSCAN found good clusters
                # Handle noise points by assigning them to nearest cluster
                if -1 in labels:
                    # Find cluster centers
                    cluster_centers = {}
                    for label in unique_labels:
                        if label != -1:
                            cluster_features = features_normalized[labels == label]
                            cluster_centers[label] = np.mean(cluster_features, axis=0)
                    
                    # Assign noise points to nearest cluster
                    for i, label in enumerate(labels):
                        if label == -1:
                            if cluster_centers:
                                distances_to_centers = {
                                    cl: np.linalg.norm(features_normalized[i] - center)
                                    for cl, center in cluster_centers.items()
                                }
                                labels[i] = min(distances_to_centers, key=distances_to_centers.get)
                            else:
                                labels[i] = 0
                
                # Remap labels to start from 0
                unique_labels = sorted(set(labels))
                label_mapping = {label: i for i, label in enumerate(unique_labels)}
                remapped_labels = [label_mapping[label] for label in labels]
                
                # Evaluate overall clustering quality with silhouette score
                if len(unique_labels) > 1:
                    try:
                        from sklearn.metrics import silhouette_score
                        silhouette = silhouette_score(features_normalized, remapped_labels)
                        print(f"DBSCAN silhouette score: {silhouette:.3f}")
                        if silhouette < 0.2:
                            print("Low silhouette score, considering as single speaker")
                            return [0] * len(features_list)
                    except Exception as sil_err:
                        print(f"Silhouette calculation failed: {sil_err}")
                
                print(f"Final result: {len(unique_labels)} speakers")
                return remapped_labels
                
        except Exception as e:
            print(f"Clustering failed: {e}")
            # Default to single speaker
            return [0] * len(features_list)

    def merge_consecutive_speakers(self, initial_speakers):
        """
        Merge consecutive segments from the same speaker, handling pauses correctly
        """
        if not initial_speakers:
            return []
        
        # Sort by start time
        initial_speakers.sort(key=lambda x: x['start'])
        
        merged_speakers = []
        current_speaker = None
        current_text_parts = []
        current_start = None
        current_end = None
        current_confidence_sum = 0
        current_count = 0
        
        for speaker_data in initial_speakers:
            speaker_id = speaker_data['speaker']
            
            # If this is the same speaker as the current one, or if it's close in time (within 3 seconds)
            if (current_speaker == speaker_id and 
                current_end is not None and 
                speaker_data['start'] - current_end <= 3000):  # 3 seconds gap tolerance
                
                # Merge with current speaker
                current_text_parts.append(speaker_data['text'])
                current_end = speaker_data['end']
                current_confidence_sum += speaker_data['confidence']
                current_count += 1
                
            else:
                # Save the previous speaker group if it exists
                if current_speaker is not None:
                    merged_text = ' '.join(current_text_parts).strip()
                    if merged_text:  # Only add if there's actual text
                        merged_speakers.append({
                            'speaker': current_speaker,
                            'text': merged_text,
                            'start': current_start,
                            'end': current_end,
                            'confidence': current_confidence_sum / current_count if current_count > 0 else 0.7
                        })
                
                # Start a new speaker group
                current_speaker = speaker_id
                current_text_parts = [speaker_data['text']]
                current_start = speaker_data['start']
                current_end = speaker_data['end']
                current_confidence_sum = speaker_data['confidence']
                current_count = 1
        
        # Don't forget the last speaker group
        if current_speaker is not None:
            merged_text = ' '.join(current_text_parts).strip()
            if merged_text:
                merged_speakers.append({
                    'speaker': current_speaker,
                    'text': merged_text,
                    'start': current_start,
                    'end': current_end,
                    'confidence': current_confidence_sum / current_count if current_count > 0 else 0.7
                })
        
        return merged_speakers

    def get_most_common_speaker(self, speakers):
        """Get the most common speaker label from a list of speaker segments"""
        if not speakers:
            return 'Speaker_1'
        
        speaker_counts = {}
        for speaker in speakers:
            speaker_counts[speaker['speaker']] = speaker_counts.get(speaker['speaker'], 0) + 1
        
        return max(speaker_counts.items(), key=lambda x: x[1])[0]

    def get_full_language_name(self, language_code):
        """Convert two-letter language code to full language name"""
        language_map = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'nl': 'Dutch',
            'pl': 'Polish',
            'ru': 'Russian',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh': 'Chinese',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'bn': 'Bengali',
            'tr': 'Turkish',
            'vi': 'Vietnamese',
            'th': 'Thai',
            'id': 'Indonesian',
            'ms': 'Malay',
            'fa': 'Persian',
            'he': 'Hebrew',
            'ur': 'Urdu',
            'el': 'Greek',
            'cs': 'Czech',
            'da': 'Danish',
            'fi': 'Finnish',
            'hu': 'Hungarian',
            'no': 'Norwegian',
            'sv': 'Swedish',
            'uk': 'Ukrainian'
        }
        return language_map.get(language_code.lower(), language_code)

    def normalize_speakers_response(self, speakers_data):
        """Convert numeric speaker IDs to alphabetical format (Speaker A, B, etc.)"""
        if not speakers_data:
            return []

        # Create mapping of numeric IDs to letters
        unique_speakers = sorted(list(set(s['speaker'] for s in speakers_data)))
        speaker_map = {
            old_id: f'Speaker {chr(65 + i)}' # A=65 in ASCII
            for i, old_id in enumerate(unique_speakers)
        }

        # Apply mapping to speakers data
        normalized_speakers = []
        for speaker in speakers_data:
            normalized_speaker = speaker.copy()
            normalized_speaker['speaker'] = speaker_map[speaker['speaker']]
            normalized_speakers.append(normalized_speaker)

        return normalized_speakers

    def pyannote_speaker_diarization(self, audio_file_path, segments):
        """
        Professional speaker diarization using pyannote.audio
        """
        try:
            
            
            
            # Initialize the pipeline (downloads model on first use)
            # Try gated model with token, else fallback to open model
            token = os.getenv("HF_AUTH_TOKEN")
            print("Token toh aaya",token)
            # Perform global login so nested models fetch with token
            if token:
                try:
                    login(token=token, add_to_git_credential=True)
                except Exception as login_err:
                    print(f"HF login warning: {login_err}")

            try:
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=token,
                )
            except Exception as gated_err:
                print(f"Gated model unavailable ({gated_err}). Falling back to public 2.1 checkpoint")
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization@2.1",
                    use_auth_token=token,
                )
            
            # Run diarization on the audio file
            diarization = pipeline(audio_file_path)
            
            # Convert diarization results to our format
            speakers_data = []
            
            for segment in segments:
                segment_start = segment['start']
                segment_end = segment['end']
                segment_middle = (segment_start + segment_end) / 2
                
                # Find which speaker is active at the middle of this segment
                speaker_label = None
                for turn, track, speaker in diarization.itertracks(yield_label=True):
                    if turn.start <= segment_middle <= turn.end:
                        speaker_label = speaker
                        break
                
                # If no speaker found, use the closest one
                if speaker_label is None:
                    min_distance = float('inf')
                    for turn, track, speaker in diarization.itertracks(yield_label=True):
                        distance = min(abs(turn.start - segment_middle), abs(turn.end - segment_middle))
                        if distance < min_distance:
                            min_distance = distance
                            speaker_label = speaker
                
                # Default to Speaker_1 if still no match
                if speaker_label is None:
                    speaker_label = "SPEAKER_00"
                
                # Format speaker name
                speaker_name = f"Speaker_{speaker_label.split('_')[-1]}" if "_" in speaker_label else "Speaker_1"
                
                speakers_data.append({
                    'speaker': speaker_name,
                    'text': segment['text'].strip(),
                    'start': int(segment['start'] * 1000),
                    'end': int(segment['end'] * 1000),
                    'confidence': 0.9
                })
            
            # Merge consecutive segments from same speaker
            merged_speakers = self.merge_consecutive_speakers_pyannote(speakers_data)
            return merged_speakers
            
        except ImportError:
            print("pyannote.audio not installed, falling back to custom diarization")
            try:
                import librosa
                audio_data, sample_rate = librosa.load(audio_file_path, sr=16000)
            except Exception:
                audio_data = None
                sample_rate = 16000
            if audio_data is not None:
                return self.advanced_speaker_diarization(audio_data, segments, sample_rate)
            else:
                # As a last resort, label all as Speaker_1
                return [{
                    'speaker': 'Speaker_1',
                    'text': segment['text'].strip(),
                    'start': int(segment['start'] * 1000),
                    'end': int(segment['end'] * 1000),
                    'confidence': 0.5
                } for segment in segments]
        except Exception as e:
            print(f"Pyannote diarization failed: {e}")
            try:
                import librosa
                audio_data, sample_rate = librosa.load(audio_file_path, sr=16000)
            except Exception:
                audio_data = None
                sample_rate = 16000
            if audio_data is not None:
                return self.advanced_speaker_diarization(audio_data, segments, sample_rate)
            else:
                return [{
                    'speaker': 'Speaker_1',
                    'text': segment['text'].strip(),
                    'start': int(segment['start'] * 1000),
                    'end': int(segment['end'] * 1000),
                    'confidence': 0.5
                } for segment in segments]

    def merge_consecutive_speakers_pyannote(self, speakers_data):
        """
        Merge consecutive segments from the same speaker for pyannote results
        """
        if not speakers_data:
            return []
        
        # Sort by start time
        speakers_data.sort(key=lambda x: x['start'])
        
        merged_speakers = []
        current_speaker = None
        current_text_parts = []
        current_start = None
        current_end = None
        
        for speaker_data in speakers_data:
            speaker_id = speaker_data['speaker']
            
            # If this is the same speaker as the current one and close in time (within 2 seconds)
            if (current_speaker == speaker_id and 
                current_end is not None and 
                speaker_data['start'] - current_end <= 2000):  # 2 seconds gap tolerance
                
                # Merge with current speaker
                current_text_parts.append(speaker_data['text'])
                current_end = speaker_data['end']
                
            else:
                # Save the previous speaker group if it exists
                if current_speaker is not None:
                    merged_text = ' '.join(current_text_parts).strip()
                    if merged_text:
                        merged_speakers.append({
                            'speaker': current_speaker,
                            'text': merged_text,
                            'start': current_start,
                            'end': current_end,
                            'confidence': 0.9
                        })
                
                # Start a new speaker group
                current_speaker = speaker_id
                current_text_parts = [speaker_data['text']]
                current_start = speaker_data['start']
                current_end = speaker_data['end']
        
        # Don't forget the last speaker group
        if current_speaker is not None:
            merged_text = ' '.join(current_text_parts).strip()
            if merged_text:
                merged_speakers.append({
                    'speaker': current_speaker,
                    'text': merged_text,
                    'start': current_start,
                    'end': current_end,
                    'confidence': 0.9
                })
        
        return merged_speakers

    def post(self, request):
        temp_file_path = None
        wav_path = None
        
        try:
            # Get the audio file from the request
            audio_file = request.FILES.get('audio')
            if not audio_file:
                return Response({'error': 'No audio file provided'}, status=status.HTTP_400_BAD_REQUEST)

            # Load Whisper model
            self.load_whisper_model()

            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                # Save uploaded file content
                for chunk in audio_file.chunks():
                    temp_file.write(chunk)
                temp_file_path = temp_file.name

            print("Processing audio file...")
            
            # Convert audio to proper format for processing
            # Load and resample to 16kHz (Whisper's expected sample rate)
            audio_data, sample_rate = librosa.load(temp_file_path, sr=16000)
            
            # Save as WAV for consistent processing
            wav_path = temp_file_path.replace('.wav', '_processed.wav')
            sf.write(wav_path, audio_data, sample_rate)

            print("Starting transcription with Whisper...")
            
            # Transcribe with Whisper (CPU optimized settings)
            whisper_result = self.whisper_model.transcribe(
                wav_path,
                language=None,  # Auto-detect language
                task='transcribe',
                verbose=False,  # Reduce output
                temperature=0.0,  # Deterministic output
                no_speech_threshold=0.6,  # Adjust for better speech detection
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.4
            )

            # Extract basic transcription
            full_text = whisper_result['text']
            detected_language = whisper_result.get('language', 'en')
            segments = whisper_result.get('segments', [])

            print(f"Transcription completed. Language: {detected_language}")

            # Prepare response with full language name
            response_data = {
                'text': full_text,
                'language': self.get_full_language_name(detected_language),
                'speakers': []
            }

            # Perform speaker diarization
            if segments:
                print("Performing speaker diarization...")
                try:
                    speakers_data = self.pyannote_speaker_diarization(wav_path, segments)
                    # Normalize speaker labels to alphabetical format
                    response_data['speakers'] = self.normalize_speakers_response(speakers_data)
                except Exception as diar_error:
                    print(f"Speaker diarization failed, using basic segmentation: {diar_error}")
                    # Fallback to basic segments with normalized speaker labels
                    basic_speakers = [{
                        'speaker': f'Speaker_{i+1}',
                        'text': segment['text'].strip(),
                        'start': int(segment['start'] * 1000),
                        'end': int(segment['end'] * 1000),
                        'confidence': 0.5
                    } for i, segment in enumerate(segments)]
                    response_data['speakers'] = self.normalize_speakers_response(basic_speakers)
            else:
                # If no segments, create one with full text
                single_speaker = [{
                    'speaker': 'Speaker_1',
                    'text': full_text,
                    'start': 0,
                    'end': int(len(audio_data) / sample_rate * 1000),
                    'confidence': 0.8
                }]
                response_data['speakers'] = self.normalize_speakers_response(single_speaker)

            print("Transcription completed successfully!")
            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            print(f"Transcription error: {str(e)}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            return Response({'error': f'Transcription failed: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        finally:
            # Clean up temporary files
            for path in [temp_file_path, wav_path]:
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except Exception as cleanup_error:
                        print(f"Warning: Could not delete temporary file {path}: {cleanup_error}")