"""Create a demo printed prescription image (with deliberate interactions) in samples/."""
from PIL import Image, ImageDraw, ImageFont


def font(size, bold=False):
    for p in [f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
              f"/usr/share/fonts/truetype/liberation/LiberationSans-{'Bold' if bold else 'Regular'}.ttf"]:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def make_rx(path="samples/prescription_demo.png"):
    W, H = 1000, 760
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 110], fill=(225, 238, 250))
    d.text((40, 22), "Dr. A. Sharma, MBBS, MD (Medicine)", fill=(20, 40, 90), font=font(30, True))
    d.text((40, 66), "City Care Clinic  |  Reg. No. 00000 (DEMO)", fill=(60, 60, 60), font=font(20))
    d.text((40, 135), "Patient: Demo Patient   Age: 58   Date: 20/11/2026", fill="black", font=font(22))
    d.text((40, 185), "Rx", fill=(20, 40, 90), font=font(40, True))
    lines = ["1. Tab Crocin 650mg 1-0-1 after food x 5 days",
             "2. Tab Combiflam SOS for pain",
             "3. Tab Warf 5mg OD x 30 days",
             "4. Tab Ecosprin 75mg HS x 30 days",
             "5. Cap Pantocid 40mg OD before breakfast x 14 days",
             "6. Tab Azithral 500mg OD x 3 days",
             "7. Tab Ondem 4mg TDS before food x 3 days"]
    y = 250
    for ln in lines:
        d.text((70, y), ln, fill="black", font=font(26))
        y += 58
    d.text((620, 690), "Signature (demo only)", fill=(90, 90, 90), font=font(18))
    im.save(path)
    return path


if __name__ == "__main__":
    print(make_rx())
