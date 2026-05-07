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

    # Part 707: Central Plate
    # Dimensions: 128x128 mm (from Schnitt A-A, 4x32 grid)
    # Thickness: PL10 (assumed)
    # Holes: 4x M24x75 (assume Ø26mm for M24)

    plate_w = 128
    plate_h = 128
    hole_dia = 26 # for M24
    hole_offset = 32 # from center lines

    # Outline
    points = [
        (0, 0),
        (plate_w, 0),
        (plate_w, plate_h),
        (0, plate_h),
        (0, 0)
    ]
    msp.add_lwpolyline(points, dxfattribs={"layer": "GEOMETRY"})

    # Holes (relative to (0,0) bottom-left corner)
    msp.add_circle((hole_offset, hole_offset), hole_dia / 2, dxfattribs={"layer": "GEOMETRY"})
    msp.add_circle((plate_w - hole_offset, hole_offset), hole_dia / 2, dxfattribs={"layer": "GEOMETRY"})
    msp.add_circle((plate_w - hole_offset, plate_h - hole_offset), hole_dia / 2, dxfattribs={"layer": "GEOMETRY"})
    msp.add_circle((hole_offset, plate_h - hole_offset), hole_dia / 2, dxfattribs={"layer": "GEOMETRY"})

    # Annotations
    msp.add_mtext("PART 707 - Central Plate", dxfattribs={"layer": "ANNOTATION"}).set_location((plate_w / 2, plate_h + 20), rotation=0, attachment_point=5)
    msp.add_mtext(f"Overall: {plate_w} x {plate_h} mm", dxfattribs={"layer": "ANNOTATION"}).set_location((plate_w / 2, plate_h + 10), rotation=0, attachment_point=5)
    msp.add_mtext("Thickness: PL10 (EST)", dxfattribs={"layer": "ANNOTATION"}).set_location((plate_w / 2, plate_h), rotation=0, attachment_point=5)
    msp.add_mtext(f"Holes: 4x Ø{hole_dia} (for M24)", dxfattribs={"layer": "ANNOTATION"}).set_location((plate_w / 2, plate_h - 10), rotation=0, attachment_point=5)

    doc.saveas(output_filename)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <output_dxf_filename>")
        sys.exit(1)
    create_part_dxf(sys.argv[1])