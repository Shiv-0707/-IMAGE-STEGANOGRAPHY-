import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image
import os
from typing import List, Tuple, Dict
import math

# ---------- Bit Helper Functions ----------
def _int_to_bits(value: int, bit_count: int) -> List[int]:
    return [(value >> (bit_count - 1 - i)) & 1 for i in range(bit_count)]

def _bits_to_int(bits: List[int]) -> int:
    value = 0
    for b in bits:
        value = (value << 1) | (b & 1)
    return value

def _text_to_bits(text: str) -> List[int]:
    b = text.encode('utf-8')
    bits = []
    for byte in b:
        bits.extend(_int_to_bits(byte, 8))
    return bits

def _bits_to_text(bits: List[int]) -> str:
    if len(bits) % 8 != 0:
        raise ValueError("Bits length not multiple of 8")
    bytes_out = bytearray()
    for i in range(0, len(bits), 8):
        byte = _bits_to_int(bits[i:i+8])
        bytes_out.append(byte)
    return bytes_out.decode('utf-8', errors='replace')

def capacity_in_bits(img: Image.Image) -> int:
    w, h = img.size
    channels = len(img.getbands())
    return w * h * channels

# ---------- Payload / Conversion Helpers ----------
def text_to_payload_bytes(text: str, encoding: str = 'utf-8') -> bytes:
    """
    payload = 4-byte big-endian length header + message bytes (UTF-8)
    """
    msg_bytes = text.encode(encoding)
    length = len(msg_bytes)
    header = length.to_bytes(4, 'big')
    return header + msg_bytes

def payload_bytes_to_bitlist(payload: bytes) -> List[int]:
    bits = []
    for b in payload:
        bits.extend([(b >> (7 - i)) & 1 for i in range(8)])
    return bits

def payload_bytes_to_hexbin(payload: bytes) -> Tuple[str, str]:
    hex_s = ' '.join(f"{b:02X}" for b in payload)
    bin_s = ' '.join(f"{b:08b}" for b in payload)
    return hex_s, bin_s

def text_to_bits_and_info(text: str, channels_per_pixel: int = 3) -> Dict:
    payload = text_to_payload_bytes(text)
    bits = payload_bytes_to_bitlist(payload)
    hex_s, bin_s = payload_bytes_to_hexbin(payload)
    total_bits = len(bits)
    pixels_needed = math.ceil(total_bits / channels_per_pixel)
    return {
        'payload_bytes': payload,
        'bit_list': bits,
        'hex': hex_s,
        'bin': bin_s,
        'total_bits': total_bits,
        'pixels_needed': pixels_needed
    }

# ---------- Core Encode / Decode ----------
def encode_text_into_image(img: Image.Image, secret_message: str) -> Image.Image:
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")

    pixels = list(img.getdata())
    channels = len(img.getbands())

    message_bytes = secret_message.encode("utf-8")
    msg_len = len(message_bytes)

    header_bits = _int_to_bits(msg_len, 32)
    message_bits = []
    for b in message_bytes:
        message_bits.extend(_int_to_bits(b, 8))
    all_bits = header_bits + message_bits

    flat = []
    for px in pixels:
        flat.extend(list(px[:channels]))

    if len(all_bits) > len(flat):
        raise ValueError("Message too large for this image!")

    for i, bit in enumerate(all_bits):
        flat[i] = (flat[i] & ~1) | bit

    new_pixels = []
    for i in range(0, len(flat), channels):
        chunk = tuple(flat[i:i+channels])
        if len(chunk) < channels:
            chunk = tuple(list(chunk) + [255] * (channels - len(chunk)))
        new_pixels.append(chunk)

    new_img = Image.new(img.mode, img.size)
    new_img.putdata(new_pixels)
    return new_img

def decode_text_from_image(img: Image.Image) -> str:
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")

    pixels = list(img.getdata())
    channels = len(img.getbands())
    flat = []
    for px in pixels:
        flat.extend(list(px[:channels]))

    header_bits = [flat[i] & 1 for i in range(32)]
    msg_len = _bits_to_int(header_bits)
    total_bits = msg_len * 8

    if 32 + total_bits > len(flat):
        raise ValueError("No valid hidden message or corrupted image!")

    message_bits = [flat[32 + i] & 1 for i in range(total_bits)]
    return _bits_to_text(message_bits)

# ---------- GUI ----------
class SteganoApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Steganography (Auto-Save, Payload Preview)")
        self.geometry("820x520")
        self.resizable(False, False)

        self.input_path = tk.StringVar()
        self.status_text = tk.StringVar(value="Select an image to start.")
        self.capacity_info = tk.StringVar(value="Capacity: N/A")

        self._build_ui()

    def _build_ui(self):
        pad = 8
        frm = ttk.Frame(self, padding=pad)
        frm.pack(fill="both", expand=True)

        # --- Input selection ---
        input_row = ttk.Frame(frm)
        input_row.pack(fill="x", pady=(0, 8))
        ttk.Label(input_row, text="Select image (any format):").pack(side="left")
        ttk.Entry(input_row, textvariable=self.input_path, width=60).pack(side="left", padx=(6, 6))
        ttk.Button(input_row, text="Browse", command=self.browse_input).pack(side="left")

        # --- Message text box ---
        ttk.Label(frm, text="Secret message:").pack(anchor="w")
        self.text_box = tk.Text(frm, height=7, wrap="word")
        self.text_box.pack(fill="x", pady=(4, 10))

        # --- Buttons ---
        btn_row = ttk.Frame(frm)
        btn_row.pack(fill="x", pady=(6, 6))
        ttk.Button(btn_row, text="Encode (Auto Save)", command=self.on_encode).pack(side="left", padx=(0, 8))
        ttk.Button(btn_row, text="Decode Message", command=self.on_decode).pack(side="left", padx=(0, 8))
        ttk.Button(btn_row, text="Show Payload", command=self.on_show_payload).pack(side="left")
        ttk.Button(btn_row, text="Clear", command=self.on_clear).pack(side="right")

        # --- Status + Info ---
        status_row = ttk.Frame(frm)
        status_row.pack(fill="x", pady=(10, 0))
        ttk.Label(status_row, textvariable=self.capacity_info).pack(side="left")
        ttk.Label(status_row, textvariable=self.status_text).pack(side="right")

        info = (
            "\nNotes:\n"
            "- You can select ANY image format (JPG, PNG, BMP, TIFF, WEBP, GIF, etc.).\n"
            "- Output automatically saved as '<originalname>_stego.png'.\n"
            "- Safe to use; PNG preserves hidden bits. JPEG may destroy LSB data.\n"
            "- 'Show Payload' reveals header+message in HEX and BIN and shows bits/pixel estimate.\n"
        )
        ttk.Label(frm, text=info, foreground="gray").pack(anchor="w", pady=(8, 0))

    # ---------- Browse ----------
    def browse_input(self):
        path = filedialog.askopenfilename(
            title="Select any image",
            filetypes=[
                ("All image formats", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.gif *.webp *.tga *.ico"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.input_path.set(path)
            try:
                img = Image.open(path)
                cap_bits = capacity_in_bits(img)
                cap_bytes = cap_bits // 8
                self.capacity_info.set(f"Capacity: {cap_bits:,} bits ({cap_bytes:,} bytes) — {img.size}, {img.mode}")
                self.status_text.set("Image loaded successfully.")
            except Exception as e:
                self.capacity_info.set("Capacity: N/A")
                self.status_text.set(f"Error loading image: {e}")

    # ---------- Encode ----------
    def on_encode(self):
        in_path = self.input_path.get().strip()
        message = self.text_box.get("1.0", "end").strip()

        if not in_path:
            messagebox.showwarning("No input", "Please select an image first.")
            return
        if message == "":
            messagebox.showwarning("Empty", "Please enter a message to hide.")
            return

        try:
            img = Image.open(in_path)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open image: {e}")
            return

        try:
            cap_bits = capacity_in_bits(img)
            needed_bits = 32 + len(message.encode("utf-8")) * 8
            if needed_bits > cap_bits:
                cap_bytes = cap_bits // 8
                messagebox.showerror("Too large", f"Message too large!\nCapacity: {cap_bytes} bytes.")
                return

            stego_img = encode_text_into_image(img, message)

            # Auto-save output beside original
            base, ext = os.path.splitext(in_path)
            out_path = f"{base}_stego.png"
            stego_img.save(out_path, format="PNG")

            self.status_text.set(f"Message embedded! Saved automatically as {out_path}")
            messagebox.showinfo("Success", f"Stego image saved automatically:\n{out_path}")

            # Also show payload (header + message) in popup after encode
            info = text_to_bits_and_info(message)
            self._show_payload_popup(info)

        except Exception as e:
            messagebox.showerror("Encoding error", f"Error during encoding:\n{e}")

    # ---------- Decode ----------
    def on_decode(self):
        path = filedialog.askopenfilename(
            title="Select image to decode",
            filetypes=[
                ("All image formats", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.gif *.webp *.tga *.ico"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        try:
            img = Image.open(path)
            message = decode_text_from_image(img)
            self.text_box.delete("1.0", "end")
            self.text_box.insert("1.0", message)
            self.status_text.set(f"Message decoded from {path}")
            messagebox.showinfo("Decoded", "Hidden message extracted successfully!")

            # Show payload built from decoded message
            info = text_to_bits_and_info(message)
            self._show_payload_popup(info, title="Decoded payload (header + message)")

        except Exception as e:
            messagebox.showerror("Decode error", f"Failed to decode message:\n{e}")

    # ---------- Show Payload ----------
    def on_show_payload(self):
        message = self.text_box.get("1.0", "end").strip()
        if message == "":
            messagebox.showwarning("Empty", "Enter a message first in the text box to view its payload.")
            return
        info = text_to_bits_and_info(message)
        self._show_payload_popup(info)

    # ---------- UI helper to display payload ----------
    def _show_payload_popup(self, info: Dict, title: str = "Payload (header + message)"):
        payload = info['payload_bytes']
        hex_s = info['hex']
        bin_s = info['bin']
        total_bits = info['total_bits']
        pixels_needed = info['pixels_needed']
        declared_len = int.from_bytes(payload[:4], 'big')

        txt = (
            f"{title}\n\n"
            f"Declared message length (bytes): {declared_len}\n"
            f"Total payload bytes (header+message): {len(payload)}\n"
            f"Total bits needed: {total_bits}\n"
            f"Estimated pixels needed (3 channels/pixel): {pixels_needed}\n\n"
            f"HEX (header + message):\n{hex_s}\n\n"
            f"BIN (header + message):\n{bin_s}\n"
        )

        # Show in scrollable popup
        popup = tk.Toplevel(self)
        popup.title(title)
        popup.geometry("700x420")
        txt_widget = tk.Text(popup, wrap="none")
        txt_widget.insert("1.0", txt)
        txt_widget.configure(state='disabled')
        txt_widget.pack(fill="both", expand=True)
        # Add simple close button
        ttk.Button(popup, text="Close", command=popup.destroy).pack(pady=6)

    # ---------- Clear ----------
    def on_clear(self):
        self.input_path.set("")
        self.text_box.delete("1.0", "end")
        self.status_text.set("Cleared.")
        self.capacity_info.set("Capacity: N/A")

# ---------- Run App ----------
if __name__ == "__main__":
    app = SteganoApp()
    app.mainloop()
