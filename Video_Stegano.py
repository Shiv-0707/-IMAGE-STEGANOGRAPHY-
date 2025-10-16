#!/usr/bin/env python3
"""
Advanced Video Steganography System
Hides data in video while maintaining full playability
"""

import os
import sys
import cv2
import numpy as np
import hashlib
import zlib
import json
import struct
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import threading
import time


class VideoSteganography:
    """Core steganography engine for video files"""

    # Magic bytes to identify steganographic content
    MAGIC_BYTES = b'VSTG'
    HEADER_SIZE = 32  # bytes for header information

    def __init__(self):
        self.bits_per_pixel = 2  # Use 2 LSBs for better capacity

    def _create_header(self, data_size: int, original_hash: bytes) -> bytes:
        """Create header with metadata"""
        header = self.MAGIC_BYTES
        header += struct.pack('>I', data_size)  # 4 bytes for size
        header += original_hash[:16]  # 16 bytes of hash
        header += b'\x00' * (self.HEADER_SIZE - len(header))  # Padding
        return header

    def _extract_header(self, header_bytes: bytes) -> Tuple[int, bytes]:
        """Extract metadata from header"""
        if not header_bytes.startswith(self.MAGIC_BYTES):
            raise ValueError("No steganographic data found (invalid magic bytes)")

        data_size = struct.unpack('>I', header_bytes[4:8])[0]
        data_hash = header_bytes[8:24]
        return data_size, data_hash

    def _embed_bits_in_pixel(self, pixel_value: int, bits: List[int]) -> int:
        """Embed bits into a pixel value using LSB"""
        # Clear the last n bits
        mask = 0xFF << self.bits_per_pixel
        pixel_value = int(pixel_value) & mask

        # Set the new bits
        for i, bit in enumerate(bits):
            if bit:
                pixel_value |= (1 << (self.bits_per_pixel - 1 - i))

        return pixel_value

    def _extract_bits_from_pixel(self, pixel_value: int) -> List[int]:
        """Extract LSB bits from a pixel"""
        bits = []
        for i in range(self.bits_per_pixel - 1, -1, -1):
            bits.append((pixel_value >> i) & 1)
        return bits

    def _bytes_to_bits(self, data: bytes) -> List[int]:
        """Convert bytes to list of bits"""
        bits = []
        for byte in data:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        return bits

    def _bits_to_bytes(self, bits: List[int]) -> bytes:
        """Convert list of bits to bytes"""
        if len(bits) % 8 != 0:
            bits.extend([0] * (8 - len(bits) % 8))

        bytes_data = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | bits[i + j]
            bytes_data.append(byte)
        return bytes(bytes_data)

    def calculate_capacity(self, video_path: str) -> int:
        """Calculate how much data can be hidden in the video (in bytes)"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Calculate total bits available (3 channels * bits_per_pixel per channel)
        total_bits = frame_count * width * height * 3 * self.bits_per_pixel
        # Convert to bytes and leave some margin for header
        return (total_bits // 8) - self.HEADER_SIZE - 100

    def embed_data(self, video_path: str, data: bytes, output_path: str,
                   progress_callback=None) -> bool:
        """Embed data into video frames"""

        # Compress data for better capacity utilization
        compressed_data = zlib.compress(data, level=9)
        data_hash = hashlib.md5(compressed_data).digest()

        # Create header
        header = self._create_header(len(compressed_data), data_hash)
        full_data = header + compressed_data

        # Convert to bits
        data_bits = self._bytes_to_bits(full_data)

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Cannot open video file")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create video writer with lossless codec
        fourcc = cv2.VideoWriter_fourcc(*'HFYU')  # HuffYUV - lossless codec
        # Try alternative codecs if HFYU not available
        if not cv2.VideoWriter(output_path, fourcc, fps, (width, height)).isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # FFV1 - another lossless codec
            if not cv2.VideoWriter(output_path, fourcc, fps, (width, height)).isOpened():
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Fallback to MJPEG (high quality)

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            cap.release()
            raise ValueError("Cannot create output video file")

        bit_index = 0
        total_bits = len(data_bits)
        frames_processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Embed bits in this frame if we still have data
            if bit_index < total_bits:
                flat_frame = frame.reshape(-1)
                pixels_needed = min((total_bits - bit_index + self.bits_per_pixel - 1) // self.bits_per_pixel,
                                    len(flat_frame))

                for pixel_idx in range(pixels_needed):
                    if bit_index >= total_bits:
                        break

                    # Get bits to embed
                    bits_to_embed = []
                    for _ in range(self.bits_per_pixel):
                        if bit_index < total_bits:
                            bits_to_embed.append(data_bits[bit_index])
                            bit_index += 1
                        else:
                            bits_to_embed.append(0)

                    # Embed bits
                    flat_frame[pixel_idx] = self._embed_bits_in_pixel(
                        flat_frame[pixel_idx], bits_to_embed
                    )

                frame = flat_frame.reshape(frame.shape)

            # Write frame
            out.write(frame)
            frames_processed += 1

            if progress_callback:
                progress = (frames_processed / frame_count) * 100
                progress_callback(progress, f"Processing frame {frames_processed}/{frame_count}")

        # Release everything
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        return True

    def extract_data(self, video_path: str, progress_callback=None) -> bytes:
        """Extract hidden data from video"""

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Cannot open video file")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # First, extract header to know data size
        header_bits = []
        total_bits_needed = -1

        frames_processed = 0
        all_bits = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            flat_frame = frame.reshape(-1)

            for pixel in flat_frame:
                bits = self._extract_bits_from_pixel(int(pixel))
                all_bits.extend(bits)

                if len(all_bits) >= self.HEADER_SIZE * 8 and len(header_bits) == 0:
                    # Extract header
                    header_bytes = self._bits_to_bytes(all_bits[:self.HEADER_SIZE * 8])
                    data_size, data_hash = self._extract_header(header_bytes)
                    total_bits_needed = (self.HEADER_SIZE + data_size) * 8
                    header_bits = all_bits[:self.HEADER_SIZE * 8]

                    if progress_callback:
                        progress_callback(0, f"Found steganographic data: {data_size} bytes")

            frames_processed += 1

            if progress_callback and frames_processed % 10 == 0:
                progress = (frames_processed / frame_count) * 100
                progress_callback(progress, f"Scanning frame {frames_processed}/{frame_count}")

            # Check if we have enough bits
            if len(header_bits) > 0 and len(all_bits) >= total_bits_needed:
                break

        cap.release()

        if len(header_bits) == 0:
            raise ValueError("No steganographic data found in video")

        # Extract the actual data
        full_data = self._bits_to_bytes(all_bits[:total_bits_needed])
        compressed_data = full_data[self.HEADER_SIZE:]

        # Verify hash
        calculated_hash = hashlib.md5(compressed_data).digest()
        if calculated_hash[:16] != data_hash:
            raise ValueError("Data integrity check failed - corrupted data")

        # Decompress and return
        try:
            original_data = zlib.decompress(compressed_data)
            return original_data
        except Exception as e:
            raise ValueError(f"Failed to decompress data - may be corrupted: {e}")


class VideoSteganoGUI:
    """GUI Application for Video Steganography"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Video Steganography System")
        self.root.geometry("900x750")

        # Set icon and style
        self.setup_styles()

        self.stego = VideoSteganography()
        self.current_video_path = None
        self.current_thread = None

        self.create_widgets()
        self.center_window()

    def setup_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')

        # Configure colors
        bg_color = '#2b2b2b'
        fg_color = '#ffffff'
        button_color = '#404040'

        self.root.configure(bg=bg_color)

        style.configure('Title.TLabel',
                        background=bg_color,
                        foreground=fg_color,
                        font=('Arial', 14, 'bold'))

        style.configure('Heading.TLabel',
                        background=bg_color,
                        foreground=fg_color,
                        font=('Arial', 11, 'bold'))

        style.configure('Info.TLabel',
                        background=bg_color,
                        foreground='#cccccc',
                        font=('Arial', 9))

    def create_widgets(self):
        """Create GUI elements"""

        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title_label = ttk.Label(main_frame,
                                text="🔒 Video Steganography System",
                                style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # Video selection frame
        video_frame = ttk.LabelFrame(main_frame, text="Select Video for Steganography", padding="10")
        video_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)

        self.video_path_var = tk.StringVar()
        ttk.Label(video_frame, text="Video File:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(video_frame, textvariable=self.video_path_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(video_frame, text="Browse", command=self.browse_video).grid(row=0, column=2)

        # Video info
        self.video_info_label = ttk.Label(video_frame, text="No video selected", style='Info.TLabel')
        self.video_info_label.grid(row=1, column=0, columnspan=3, pady=(5, 0))

        # Standalone Video Player frame
        player_frame = ttk.LabelFrame(main_frame, text="Standalone Video Player", padding="10")
        player_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        ttk.Button(player_frame, text="Browse and Play Video File", command=self.play_selected_video).pack(pady=5)

        # Operation tabs
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)

        # Embed tab
        embed_frame = ttk.Frame(notebook, padding="10")
        notebook.add(embed_frame, text="Hide Data")

        ttk.Label(embed_frame, text="Data to Hide:", style='Heading.TLabel').grid(row=0, column=0, sticky=tk.W)

        # Data type selection
        self.data_type = tk.StringVar(value="text")
        ttk.Radiobutton(embed_frame, text="Text Message",
                        variable=self.data_type, value="text",
                        command=self.toggle_data_input).grid(row=1, column=0, sticky=tk.W)
        ttk.Radiobutton(embed_frame, text="File",
                        variable=self.data_type, value="file",
                        command=self.toggle_data_input).grid(row=1, column=1, sticky=tk.W)

        # Text input
        self.text_frame = ttk.Frame(embed_frame)
        self.text_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        self.text_input = scrolledtext.ScrolledText(self.text_frame, height=10, width=60)
        self.text_input.pack()

        # File input
        self.file_frame = ttk.Frame(embed_frame)

        self.file_path_var = tk.StringVar()
        ttk.Label(self.file_frame, text="File:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(self.file_frame, textvariable=self.file_path_var, width=40).grid(row=0, column=1, padx=5)
        ttk.Button(self.file_frame, text="Browse", command=self.browse_file).grid(row=0, column=2)

        # Embed button
        self.embed_button = ttk.Button(embed_frame, text="Hide Data in Video",
                                       command=self.embed_data,
                                       state='disabled')
        self.embed_button.grid(row=4, column=0, columnspan=2, pady=20)

        # Extract tab
        extract_frame = ttk.Frame(notebook, padding="10")
        notebook.add(extract_frame, text="Extract Data")

        ttk.Label(extract_frame, text="Extract hidden data from video",
                  style='Heading.TLabel').grid(row=0, column=0, pady=10)

        self.extract_button = ttk.Button(extract_frame, text="Extract Hidden Data",
                                         command=self.extract_data,
                                         state='disabled')
        self.extract_button.grid(row=1, column=0, pady=10)

        # Results area
        ttk.Label(extract_frame, text="Extracted Data:").grid(row=2, column=0, sticky=tk.W, pady=(20, 5))

        self.result_text = scrolledtext.ScrolledText(extract_frame, height=10, width=60)
        self.result_text.grid(row=3, column=0, pady=5)

        ttk.Button(extract_frame, text="Save Extracted Data",
                   command=self.save_extracted_data).grid(row=4, column=0, pady=10)

        # Video preview tab
        preview_frame = ttk.Frame(notebook, padding="10")
        notebook.add(preview_frame, text="Preview")

        self.preview_label = ttk.Label(preview_frame, text="Video preview is not available in this version.\nUse the 'Standalone Video Player' to play videos.")
        self.preview_label.pack()

        ttk.Button(preview_frame, text="Play Original Loaded Video",
                   command=lambda: self.play_video('original')).pack(side=tk.LEFT, padx=5, pady=10)
        ttk.Button(preview_frame, text="How to Play Stego Video?",
                   command=lambda: self.play_video('stego')).pack(side=tk.LEFT, padx=5, pady=10)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame,
                                            variable=self.progress_var,
                                            maximum=100)
        self.progress_bar.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        # Status bar
        self.status_label = ttk.Label(main_frame, text="Ready", style='Info.TLabel')
        self.status_label.grid(row=5, column=0, columnspan=3, sticky=tk.W)

        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)

    def center_window(self):
        """Center the window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def toggle_data_input(self):
        """Toggle between text and file input"""
        if self.data_type.get() == "text":
            self.file_frame.grid_forget()
            self.text_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        else:
            self.text_frame.grid_forget()
            self.file_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

    def browse_video(self):
        """Browse for video file"""
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv"),
                ("All files", "*.*")
            ]
        )

        if filename:
            self.video_path_var.set(filename)
            self.current_video_path = filename
            self.analyze_video(filename)

    def analyze_video(self, video_path):
        """Analyze video and display information"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Cannot open video")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0

            cap.release()

            # Calculate capacity
            capacity = self.stego.calculate_capacity(video_path)
            capacity_mb = capacity / (1024 * 1024)

            info_text = (f"Resolution: {width}x{height} | "
                         f"FPS: {fps:.2f} | "
                         f"Frames: {frame_count} | "
                         f"Duration: {duration:.2f}s | "
                         f"Capacity: {capacity_mb:.2f} MB")

            self.video_info_label.config(text=info_text)

            # Enable buttons
            self.embed_button.config(state='normal')
            self.extract_button.config(state='normal')

            self.status_label.config(text=f"Video loaded: {os.path.basename(video_path)}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze video: {str(e)}")
            self.video_info_label.config(text="Error analyzing video")

    def browse_file(self):
        """Browse for file to hide"""
        filename = filedialog.askopenfilename(
            title="Select File to Hide",
            filetypes=[("All files", "*.*")]
        )

        if filename:
            self.file_path_var.set(filename)

            # Check file size
            file_size = os.path.getsize(filename)
            if self.current_video_path:
                capacity = self.stego.calculate_capacity(self.current_video_path)
                if file_size > capacity:
                    messagebox.showwarning(
                        "File Too Large",
                        f"File size ({file_size / 1024:.2f} KB) exceeds "
                        f"video capacity ({capacity / 1024:.2f} KB)"
                    )

    def update_progress(self, value, message):
        """Update progress bar and status"""
        self.progress_var.set(value)
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def embed_data(self):
        """Embed data into video"""
        if not self.current_video_path:
            messagebox.showerror("Error", "Please select a video first")
            return

        # Get data to embed
        if self.data_type.get() == "text":
            text = self.text_input.get("1.0", tk.END).strip()
            if not text:
                messagebox.showerror("Error", "Please enter text to hide")
                return
            data = text.encode('utf-8')
        else:
            file_path = self.file_path_var.get()
            if not file_path or not os.path.exists(file_path):
                messagebox.showerror("Error", "Please select a file to hide")
                return

            with open(file_path, 'rb') as f:
                file_data = f.read()

            # Create metadata
            metadata = {
                'filename': os.path.basename(file_path),
                'size': len(file_data)
            }
            metadata_json = json.dumps(metadata).encode('utf-8')

            # Combine metadata and file data
            data = struct.pack('>I', len(metadata_json)) + metadata_json + file_data

        # Check capacity
        capacity = self.stego.calculate_capacity(self.current_video_path)
        if len(data) > capacity:
            messagebox.showerror(
                "Data Too Large",
                f"Data size ({len(data) / 1024:.2f} KB) exceeds "
                f"video capacity ({capacity / 1024:.2f} KB)"
            )
            return

        # Get output path
        output_path = filedialog.asksaveasfilename(
            title="Save Stego Video As",
            defaultextension=".avi",
            filetypes=[
                ("AVI files", "*.avi"),
                ("MP4 files", "*.mp4"),
                ("All files", "*.*")
            ]
        )

        if not output_path:
            return

        # Embed in thread
        def embed_thread():
            try:
                self.embed_button.config(state='disabled')
                self.extract_button.config(state='disabled')

                success = self.stego.embed_data(
                    self.current_video_path,
                    data,
                    output_path,
                    progress_callback=self.update_progress
                )

                if success:
                    self.root.after(0, lambda: messagebox.showinfo(
                        "Success",
                        f"Data successfully hidden in video!\n"
                        f"Output: {output_path}\n"
                        f"Data size: {len(data) / 1024:.2f} KB"
                    ))
                    self.root.after(0, lambda: self.status_label.config(
                        text=f"Successfully created stego video: {os.path.basename(output_path)}"
                    ))

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Error",
                    f"Failed to embed data: {str(e)}"
                ))
            finally:
                self.root.after(0, lambda: self.embed_button.config(state='normal'))
                self.root.after(0, lambda: self.extract_button.config(state='normal'))
                self.root.after(0, lambda: self.progress_var.set(0))

        self.current_thread = threading.Thread(target=embed_thread, daemon=True)
        self.current_thread.start()

    def extract_data(self):
        """Extract hidden data from video"""
        if not self.current_video_path:
            messagebox.showerror("Error", "Please select a video first")
            return

        # Extract in thread
        def extract_thread():
            try:
                self.embed_button.config(state='disabled')
                self.extract_button.config(state='disabled')

                data = self.stego.extract_data(
                    self.current_video_path,
                    progress_callback=self.update_progress
                )

                # Check if it's a file
                try:
                    metadata_len = struct.unpack('>I', data[:4])[0]
                    metadata_json = data[4:4 + metadata_len]
                    metadata = json.loads(metadata_json.decode('utf-8'))

                    if 'filename' in metadata:
                        # It's a file
                        file_data = data[4 + metadata_len:]

                        self.root.after(0, lambda: self.result_text.delete("1.0", tk.END))
                        self.root.after(0, lambda: self.result_text.insert(
                            "1.0",
                            f"Extracted File:\n"
                            f"Name: {metadata['filename']}\n"
                            f"Size: {metadata['size'] / 1024:.2f} KB\n\n"
                            f"Use 'Save Extracted Data' to save the file."
                        ))

                        # Store for saving
                        self.extracted_file_data = file_data
                        self.extracted_file_name = metadata['filename']
                    else:
                        raise ValueError("Not a file")

                except:
                    # It's text
                    text = data.decode('utf-8', errors='replace')
                    self.root.after(0, lambda: self.result_text.delete("1.0", tk.END))
                    self.root.after(0, lambda: self.result_text.insert("1.0", text))
                    self.extracted_file_data = None

                self.root.after(0, lambda: messagebox.showinfo(
                    "Success",
                    f"Data successfully extracted!\n"
                    f"Size: {len(data) / 1024:.2f} KB"
                ))
                self.root.after(0, lambda: self.status_label.config(
                    text="Data extraction successful"
                ))

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Error",
                    f"Failed to extract data: {str(e)}\n\n"
                    f"This video may not contain hidden data."
                ))
                self.root.after(0, lambda: self.result_text.delete("1.0", tk.END))
                self.root.after(0, lambda: self.result_text.insert(
                    "1.0",
                    "No hidden data found or extraction failed."
                ))
            finally:
                self.root.after(0, lambda: self.embed_button.config(state='normal'))
                self.root.after(0, lambda: self.extract_button.config(state='normal'))
                self.root.after(0, lambda: self.progress_var.set(0))

        self.current_thread = threading.Thread(target=extract_thread, daemon=True)
        self.current_thread.start()

    def save_extracted_data(self):
        """Save extracted data to file"""
        content = self.result_text.get("1.0", tk.END).strip()

        if not content or content == "No hidden data found or extraction failed.":
            messagebox.showwarning("Warning", "No data to save")
            return

        if hasattr(self, 'extracted_file_data') and self.extracted_file_data:
            # Save as original file
            filename = filedialog.asksaveasfilename(
                title="Save Extracted File",
                initialfile=self.extracted_file_name,
                filetypes=[("All files", "*.*")]
            )

            if filename:
                with open(filename, 'wb') as f:
                    f.write(self.extracted_file_data)
                messagebox.showinfo("Success", f"File saved: {filename}")
        else:
            # Save as text
            filename = filedialog.asksaveasfilename(
                title="Save Extracted Text",
                defaultextension=".txt",
                filetypes=[
                    ("Text files", "*.txt"),
                    ("All files", "*.*")
                ]
            )

            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                messagebox.showinfo("Success", f"Text saved: {filename}")

    def _open_file_with_default_app(self, file_path: str):
        """Opens a file with the system's default application in a cross-platform way."""
        try:
            if sys.platform == 'win32':
                os.startfile(file_path)
            elif sys.platform == 'darwin':  # macOS
                subprocess.run(['open', file_path], check=True)
            else:  # linux and other UNIX-like
                subprocess.run(['xdg-open', file_path], check=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            messagebox.showerror("Error", f"Could not open file: {e}\n"
                                          f"Please ensure you have a default application for this file type.")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred while opening the file: {e}")

    def _play_video_smartly(self, video_path: str):
        """Plays a video, intelligently choosing ffplay for problematic formats like .avi and .mkv."""
        if not video_path or not os.path.exists(video_path):
            messagebox.showerror("Error", f"Video file not found:\n{video_path}")
            return

        _, extension = os.path.splitext(video_path)

        # Use ffplay for formats that default players often struggle with
        if extension.lower() in ['.mkv', '.avi']:
            try:
                self.status_label.config(text=f"Attempting to play {os.path.basename(video_path)} with ffplay...")
                # Use Popen to not block the GUI
                subprocess.Popen(['ffplay', '-i', video_path])
            except FileNotFoundError:
                messagebox.showerror(
                    "FFmpeg Not Found",
                    "Could not find 'ffplay'.\n\n"
                    "Please install FFmpeg and ensure 'ffplay' is in your system's PATH to play .mkv and .avi files.\n\n"
                    "Windows: Download from https://ffmpeg.org/download.html\n"
                    "macOS: Install with 'brew install ffmpeg'\n"
                    "Linux: Install with 'sudo apt install ffmpeg' or equivalent"
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to play video with ffplay: {e}")
        else:
            self.status_label.config(text=f"Opening {os.path.basename(video_path)} with default player...")
            self._open_file_with_default_app(video_path)

    def play_selected_video(self):
        """Browse and play a video file using the smart playback logic."""
        video_path = filedialog.askopenfilename(
            title="Select Video to Play",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv"),
                ("All files", "*.*")
            ]
        )
        if video_path:
            self._play_video_smartly(video_path)

    def play_video(self, video_type):
        """Play the original video loaded into the application."""
        if video_type == 'original':
            if self.current_video_path:
                self._play_video_smartly(self.current_video_path)
            else:
                messagebox.showinfo("Info", "No video loaded yet. Please select a video for steganography first.")
        elif video_type == 'stego':
            messagebox.showinfo("Info", "To play the stego video you created, please use the "
                                        "'Browse and Play Video File' button and select your saved file.")

    def run(self):
        """Run the application"""
        self.root.mainloop()


def main():
    """Main entry point"""
    print("=" * 60)
    print("Video Steganography System")
    print("Hide data in videos while keeping them playable")
    print("=" * 60)

    # Check dependencies
    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
    except ImportError:
        print("ERROR: OpenCV not installed!")
        print("Install with: pip install opencv-python")
        sys.exit(1)

    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
    except ImportError:
        print("ERROR: NumPy not installed!")
        print("Install with: pip install numpy")
        sys.exit(1)

    try:
        from PIL import Image
        print("PIL/Pillow: OK")
    except ImportError:
        print("ERROR: Pillow not installed!")
        print("Install with: pip install Pillow")
        sys.exit(1)

    print("-" * 60)
    print("NOTE: For playing .mkv and .avi files, this application uses 'ffplay'.")
    print("Please ensure you have FFmpeg installed and added to your system's PATH.")
    print("Download FFmpeg from: https://ffmpeg.org/download.html")
    print("=" * 60)
    print("Starting GUI...")

    app = VideoSteganoGUI()
    app.run()


if __name__ == "__main__":
    main()