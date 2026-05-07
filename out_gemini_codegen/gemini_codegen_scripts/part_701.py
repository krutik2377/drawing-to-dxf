import sys
import ezdxf
import math

def create_dxf_part_701(out_file):
    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()

    doc.layers.add("GEOMETRY", color=1) # Red
    doc.layers.add("DIMENSION", color=3) # Green
    doc.layers.add("ANNOTATION", color=7) # White

    # Part 701: Steigbügel (Stirrup)
    # Material: PL 20x28 (from detail view)
    # Overall dimensions: Length 477 + 125 = 602mm, Width 250mm
    # Outline: Rectangle 477x250 with a semi-circular end of R125 on the right
    
    length_straight = 477
    width = 250
    radius_end = 125
    thickness = 20
    
    # Outline as LWPOLYLINE with an arc segment
    # Start at (0,0), go right, then arc up, then left, then down
    points_outline = [
        (0, 0), # Start bottom-left
        (length_straight, 0) # Go right to start of arc
    ]
    # Add arc segment: from (length_straight, 0) to (length_straight, width), center (length_straight, width/2)
    # ezdxf LWPOLYLINE arc requires bulge value. Bulge for 180 deg arc is (tan(angle/4)). Angle = 180 deg = pi rad. angle/4 = pi/4. tan(pi/4) = 1.
    # For a semi-circle from (x,y1) to (x,y2) with center (x, (y1+y2)/2), bulge is 1 or -1 depending on direction.
    # From (length_straight, 0) to (length_straight, width) in counter-clockwise direction, bulge is 1.
    msp.add_lwpolyline([
        (0, 0),
        (length_straight, 0, 1), # Point (length_straight,0) with bulge 1 for CCW arc to (length_straight, width)
        (length_straight, width),
        (0, width),
        (0, 0)
    ], dxfattribs={'layer': 'GEOMETRY'})

    # Holes
    # M35 hole: Ø37
    msp.add_circle((125, 125), 37/2, dxfattribs={'layer': 'GEOMETRY'})
    
    # 6x M24 holes: Ø26
    hole_radius_m24 = 26/2
    holes_m24_coords = [
        (125, 50), (125, 200),
        (300, 50), (300, 200),
        (390, 50), (390, 200)
    ]
    for x, y in holes_m24_coords:
        msp.add_circle((x, y), hole_radius_m24, dxfattribs={'layer': 'GEOMETRY'})

    # Annotations
    msp.add_mtext("PART 701 - Steigbügel", dxfattribs={'layer': 'ANNOTATION'}).set_location((10, width + 10), 5, 5)
    msp.add_mtext(f"MATERIAL: PL {thickness}x28 (EST from detail)", dxfattribs={'layer': 'ANNOTATION'}).set_location((10, width + 20), 5, 5)
    msp.add_mtext(f"OVERALL: L {length_straight + radius_end} x W {width} mm", dxfattribs={'layer': 'ANNOTATION'}).set_location((10, width + 30), 5, 5)
    msp.add_mtext(f"1x Ø37 (M35)", dxfattribs={'layer': 'ANNOTATION'}).set_location((125, 125 + 20), 5, 5)
    msp.add_mtext(f"6x Ø26 (M24)", dxfattribs={'layer': 'ANNOTATION'}).set_location((10, width + 40), 5, 5)

    doc.saveas(out_file)

if __name__ == '__main__':
    create_dxf_part_701(sys.argv[1])