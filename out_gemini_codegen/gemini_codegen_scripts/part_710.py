import sys
import ezdxf
import math

def create_dxf_part_710(out_file):
    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()

    doc.layers.add("GEOMETRY", color=1)
    doc.layers.add("DIMENSION", color=3)
    doc.layers.add("ANNOTATION", color=7)

    # Part 710: L-Section Member
    # Section: L 150x50x10 (unequal angle)
    # Segment length: EST 500mm for representation
    
    leg_long = 150
    leg_short = 50
    thickness = 10
    segment_length = 500 # EST length for drawing
    
    # Outline of L-section (unequal angle)
    # Assuming the 150mm leg is vertical (along Y-axis) and 50mm leg is horizontal (along X-axis)
    # (0,0) is the inner corner.
    points_L_section = [
        (0, leg_long), # Top-left of vertical leg
        (0, 0),        # Bottom-left (inner corner)
        (leg_short, 0), # Bottom-right of horizontal leg
        (leg_short, thickness), # Inner corner of horizontal leg
        (thickness, thickness), # Inner corner of vertical leg
        (thickness, leg_long), # Top-right of vertical leg
        (0, leg_long)  # Close to top-left
    ]
    # To draw a segment, we extend the horizontal leg along the segment_length
    points_segment = [
        (0, leg_long),
        (0, 0),
        (segment_length, 0),
        (segment_length, thickness),
        (thickness, thickness),
        (thickness, leg_long),
        (0, leg_long)
    ]
    msp.add_lwpolyline(points_segment, dxfattribs={'layer': 'GEOMETRY'})

    # Holes: M20x50 (Ø22) for 739
    # 4x M20 holes. Gauge 75mm from the 150mm leg (EST).
    # Spacing: EST 50mm.
    hole_radius_m20 = 22/2
    gauge_line = 75 # Gauge for 150mm leg
    
    x_coords_m20 = [50, 100, 150, 200] # EST spacing 50mm
    for x in x_coords_m20:
        msp.add_circle((x, gauge_line), hole_radius_m20, dxfattribs={'layer': 'GEOMETRY'})

    # Annotations
    msp.add_mtext("PART 710 - L-Section Member", dxfattribs={'layer': 'ANNOTATION'}).set_location((10, leg_long + 10), 5, 5)
    msp.add_mtext(f"SECTION: L {leg_long}x{leg_short}x{thickness}", dxfattribs={'layer': 'ANNOTATION'}).set_location((10, leg_long + 20), 5, 5)
    msp.add_mtext(f"LENGTH: EST {segment_length} mm (segment)", dxfattribs={'layer': 'ANNOTATION'}).set_location((10, leg_long + 30), 5, 5)
    msp.add_mtext(f"4x Ø22 (M20)", dxfattribs={'layer': 'ANNOTATION'}).set_location((10, leg_long + 40), 5, 5)

    doc.saveas(out_file)

if __name__ == '__main__':
    create_dxf_part_710(sys.argv[1])