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

    # Part 731: Angle Member
    # Section: L 90x90x9 (ISA 90x90x9)
    # Holes: M24x75 (assume Ø26mm for M24)
    # Drawing a segment of 300mm length with typical hole pattern

    leg_size = 90
    thickness = 9
    segment_length = 300 # Arbitrary segment length for representation
    hole_dia = 26 # for M24
    hole_offset_from_corner = 32 # from Schnitt A-A
    hole_spacing = 75 # from main elevation

    # Outline of the L-profile (top view of one leg)
    # We'll draw one leg as a flat bar for simplicity in 2D.
    # The Schnitt A-A shows the cross-section. We'll draw a flat projection.

    # Draw the main leg (e.g., the horizontal one in Schnitt A-A)
    points = [
        (0, 0),
        (segment_length, 0),
        (segment_length, leg_size),
        (0, leg_size),
        (0, 0)
    ]
    msp.add_lwpolyline(points, dxfattribs={"layer": "GEOMETRY"})

    # Holes on this leg
    # Holes are 32mm from the edge (which is 0 or leg_size)
    # Let's place them 32mm from the 'bottom' edge (y=0)
    # And at 75mm, 150mm, 225mm along the length
    hole_y_pos = hole_offset_from_corner
    msp.add_circle((hole_spacing, hole_y_pos), hole_dia / 2, dxfattribs={"layer": "GEOMETRY"})
    msp.add_circle((hole_spacing * 2, hole_y_pos), hole_dia / 2, dxfattribs={"layer": "GEOMETRY"})
    msp.add_circle((hole_spacing * 3, hole_y_pos), hole_dia / 2, dxfattribs={"layer": "GEOMETRY"})

    # Annotations
    msp.add_mtext("PART 731 - Angle Member", dxfattribs={"layer": "ANNOTATION"}).set_location((segment_length / 2, leg_size + 20), rotation=0, attachment_point=5)
    msp.add_mtext(f"Section: ISA {leg_size}x{leg_size}x{thickness} mm", dxfattribs={"layer": "ANNOTATION"}).set_location((segment_length / 2, leg_size + 10), rotation=0, attachment_point=5)
    msp.add_mtext(f"Holes: Ø{hole_dia} (for M24) @ {hole_offset_from_corner} from edge, {hole_spacing} spacing", dxfattribs={"layer": "ANNOTATION"}).set_location((segment_length / 2, leg_size), rotation=0, attachment_point=5)
    msp.add_mtext(f"Note: Shown as a {segment_length}mm segment", dxfattribs={"layer": "ANNOTATION"}).set_location((segment_length / 2, leg_size - 10), rotation=0, attachment_point=5)

    doc.saveas(output_filename)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <output_dxf_filename>")
        sys.exit(1)
    create_part_dxf(sys.argv[1])