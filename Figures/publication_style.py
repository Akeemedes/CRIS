"""Shared print-size typography for the manuscript figures."""
from matplotlib.text import Text

MIN_FONT_PT = 11

def enforce_print_fonts(figure):
    """Preserve larger text and prevent local overrides from shrinking labels."""
    for axis in figure.axes:
        axis.tick_params(axis='both', which='both', labelsize=MIN_FONT_PT)
        for artist in (axis.xaxis.label, axis.yaxis.label):
            artist.set_fontsize(max(12, artist.get_fontsize()))
    for artist in figure.findobj(Text):
        artist.set_fontsize(max(MIN_FONT_PT, artist.get_fontsize()))
