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

    # Part 721 (and 722): Base Plate / Flat Bar
    # Dimensions: L 250x28 (Length 250mm, Width 28mm, Thickness 28mm)
    # Holes: 2x M27x85 (assume Ø29mm for M27)

    length = 250
    width = 28
    thickness = 28
    hole_dia = 29 # for M27
    hole_offset_end = 85

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
    msp.add_mtext("PART 721 - Base Plate", dxfattribs={"layer": "ANNOTATION"}).set_location((length / 2, width + 20), rotation=0, attachment_point=5)
    msp.add_mtext("QTY: 2 (also covers Part 722)", dxfattribs={"layer": "ANNOTATION"}).set_location((length / 2, width + 10), rotation=0, attachment_point=5)
    msp.add_mtext(f"Section: FL {length}x{width}x{thickness} mm", dxfattribs={"layer": "ANNOTATION"}).set_location((length / 2, width), rotation=0, attachment_point=5)
    msp.add_mtext(f"Holes: 2x Ø{hole_dia} (for M27) @ {hole_offset_end} from ends", dxfattribs={"layer": "ANNOTATION"}).set_location((length / 2, width - 10), rotation=0, attachment_point=5)

    doc.saveas(output_filename)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <output_dxf_filename>")
        sys.exit(1)
    create_part_dxf(sys.argv[1])