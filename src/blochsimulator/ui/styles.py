"""Small cross-platform Qt style fragments shared by desktop workspaces."""

# macOS' native QGroupBox style does not consistently honor a font declared
# only on QGroupBox::title.  Bold the group itself (which paints the title),
# then explicitly return contained widgets to normal weight.  Nested group
# boxes receive bold again for their own titles.
BOLD_GROUP_TITLES_STYLE = (
    "QGroupBox { font-weight: bold; }"
    "QGroupBox QWidget { font-weight: normal; }"
    "QGroupBox QGroupBox { font-weight: bold; }"
)
