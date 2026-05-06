"""Single-file layout DXF for Autodesk-style viewers."""

from drawing_to_dxf.export_dxf import export_viewer_layout_dxf
from drawing_to_dxf.link_parts import PartGroup
from drawing_to_dxf.vectorize import Segment


def test_export_viewer_layout_writes_layers(tmp_path) -> None:
    import ezdxf

    g = PartGroup(
        part_id="p0",
        label_center=(10.0, 10.0),
        label_box_pad=(0, 0, 0, 0),
        segments=[Segment(0, 5, 20, 5)],
    )
    outp = tmp_path / "layout.dxf"
    export_viewer_layout_dxf(outp, [("PANEL_00", g, 40.0, 30.0)], mm_per_pixel=1.0, gap_mm=5.0)
    assert outp.is_file()
    doc = ezdxf.readfile(str(outp))
    assert doc.layers.has_entry("PANEL_00")
