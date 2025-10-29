import qrcode
import os

def generate_qr_code(user_id: str, qr_format="USER_ID|{}"):
    """
    Generates a QR code image file storing the formatted user ID.
    Returns the exact string data encoded in the QR code.
    """
    qr_data = qr_format.format(user_id)
    filename = f"qr_{user_id}.png"

    # Create the QR code image
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4)
    qr.add_data(qr_data)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Ensure a directory for QR codes exists
    qr_dir = "qrcodes"
    os.makedirs(qr_dir, exist_ok=True)
    full_path = os.path.join(qr_dir, filename)
    
    img.save(full_path)
    print(f"INFO: Generated QR code image saved as {full_path}")
    return qr_data

if __name__ == '__main__':
    print("--- QR CODE GENERATION ---")
    
    # Define the users to generate QR codes for
    safe_id = "utsav16"
    mixed_id = "aarnov321"
    scammer_id = "himansu367"

    # Generate QR codes for all users
    generate_qr_code(safe_id)
    generate_qr_code(mixed_id)
    generate_qr_code(scammer_id)

    print("Generation complete. Run 'python api.py' or 'python fraud_scanner.py' next.")
