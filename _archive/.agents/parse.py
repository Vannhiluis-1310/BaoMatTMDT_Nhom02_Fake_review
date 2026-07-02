import os
import shutil

src = r"c:\Users\vanhi\Desktop\HCMUTE_TMDT\HKII_Nam_3\Bao_Mat_TMDT\Fake_reviews\.agents\ORIGINAL_REQUEST.md"
dst = r"c:\Users\vanhi\Desktop\HCMUTE_TMDT\HKII_Nam_3\Bao_Mat_TMDT\Fake_reviews\.agents\parsed_request.md"

with open(src, 'r', encoding='utf-8') as f:
    content = f.read()

with open(dst, 'w', encoding='utf-8') as f:
    f.write(content)
