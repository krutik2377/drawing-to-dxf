import sys
import ezdxf
import math

def create_part_dxf(output_filename):
    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()

    # Layers
    doc.layers.add("GEOMETRY", color=7)
    doc.layers.add("DIMENSION", color=3)
    doc.layers.add("ANNOTATION", color=1)

    # Part 739: Horizontal Member / Flat Bar
    # Dimensions: L 150x15 (Length 150mm, Width 15mm, Thickness 15mm)
    # Holes: M20x50 (assume Ø22mm for M20) - 2 holes, one at each end

    length = 150
    width = 15
    thickness = 15
    hole_dia = 22 # for M20
    hole_offset_end = 50

    # Outline
    points = [
        (0, 0),
        (length, 0),
        (length, width),
        (0, width),
        (0, 0)
    ]
    msp.add_lwpolyline(points, dxfattribs={"layer": "GEOMETRY"})

    # Holes (centered along width)
    msp.add_circle((hole_offset_end, width / 2), hole_dia / 2, dxfattribs={"layer": "GEOMETRY"})
    msp.add_circle((length - hole_offset_end, width / 2), hole_dia / 2, dxfattribs={"layer": "GEOMETRY"})

    # Annotations
    msp.add_mtext("PART 739 - Horizontal Member", dxfattribs={"layer": "ANNOTATION"}).set_location((length / 2, width + 20), rotation=0, attachment_point=5)
    msp.add_mtext(f"Section: FL {length}x{width}x{thickness} mm", dxfattribs={"layer": "ANNOTATION"}).set_location((length / 2, width + 10), rotation=0, attachment_point=5)
    msp.add_mtext(f"Holes: 2x Ø{hole_dia} (for M20) @ {hole_offset_end} from ends", dxfattribs={"layer": "ANNOTATION"}).set_location((length / 2, width), rotation=0, attachment_point=5)

    doc.saveas(output_filename)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <output_dxf_filename>")
        sys.exit(1)
    create_part_dxf(sys.argv[1])