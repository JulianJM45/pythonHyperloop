# %%

from matplotlib import font_manager

for f in font_manager.fontManager.ttflist:
    # if "NewCM" in f.name:
    print(f.name)
