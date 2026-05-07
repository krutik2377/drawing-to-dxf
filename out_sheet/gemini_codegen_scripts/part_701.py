import sys
import ezdxf
import math

def draw_slotted_hole(msp, center_x, center_y, slot_length, hole_diameter, angle_deg, layer="GEOMETRY"):
    radius = hole_diameter / 2
    half_slot_length_center_to_center = (slot_length - hole_diameter) / 2

    angle_rad = math.radians(angle_deg)

    # Calculate end points of the slot's center line
    p1_x = center_x - half_slot_length_center_to_center * math.cos(angle_rad)
    p1_y = center_y - half_slot_length_center_to_center * math.sin(angle_rad)
    p2_x = center_x + half_slot_length_center_to_center * math.cos(angle_rad)
    p2_y = center_y + half_slot_length_center_to_center * math.sin(angle_rad)

    # Add end circles
    msp.add_circle((p1_x, p1_y), radius, dxfattribs={"layer": layer})
    msp.add_circle((p2_x, p2_y), radius, dxfattribs={"layer": layer})

    # Add connecting lines (tangents)
    # Perpendicular angle for tangents
    perp_angle_rad = angle_rad + math.pi / 2

    # Top line
    msp.add_line(
        (p1_x + radius * math.cos(perp_angle_rad), p1_y + radius * math.sin(perp_angle_rad)),
        (p2_x + radius * math.cos(perp_angle_rad), p2_y + radius * math.sin(perp_angle_rad)),
        dxfattribs={"layer": layer}
    )
    # Bottom line
    msp.add_line(
        (p1_x - radius * math.cos(perp_angle_rad), p1_y - radius * math.sin(perp_angle_rad)),
        (p2_x - radius * math.cos(perp_angle_rad), p2_y - radius * math.sin(perp_angle_rad)),
        dxfattribs={"layer": layer}
    )

def create_part_dxf(output_filename):
    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()

    # Layers
    doc.layers.add("GEOMETRY", color=7)
    doc.layers.add("DIMENSION", color=3)
    doc.layers.add("ANNOTATION", color=1)

    # Part 701: Steigbügel (Stirrup)
    # Overall dimensions: 375 (width) x 437 (height)
    # Central square: 50x50
    # Arm width: 50
    # Chamfer: 10x10
    # Fillets: R12 (inner), R20 (outer)

    half_width = 375 / 2
    half_height = 437 / 2
    central_half_side = 50 / 2
    chamfer_size = 10
    R12 = 12
    R20 = 20
    bulge_90_deg_cw = -math.tan(math.radians(45/2)) # For clockwise arc

    # Outer profile (clockwise)
    outer_points = [
        (half_width - chamfer_size, central_half_side, 0), # 1
        (half_width, central_half_side - chamfer_size, 0), # 2
        (half_width, -central_half_side + chamfer_size, 0), # 3
        (half_width - chamfer_size, -central_half_side, 0), # 4
        (central_half_side + R20, -central_half_side, 0), # 5
        (central_half_side, -central_half_side - R20, bulge_90_deg_cw), # 6 (R20 fillet)
        (central_half_side, -half_height + chamfer_size, 0), # 7
        (central_half_side + chamfer_size, -half_height, 0), # 8
        (-central_half_side - chamfer_size, -half_height, 0), # 9
        (-central_half_side, -half_height + chamfer_size, 0), # 10
        (-central_half_side, -central_half_side - R20, bulge_90_deg_cw), # 11 (R20 fillet)
        (-central_half_side - R20, -central_half_side, 0), # 12
        (-half_width + chamfer_size, -central_half_side, 0), # 13
        (-half_width, -central_half_side + chamfer_size, 0), # 14
        (-half_width, central_half_side - chamfer_size, 0), # 15
        (-half_width + chamfer_size, central_half_side, 0), # 16
        (-central_half_side - R20, central_half_side, 0), # 17
        (-central_half_side, central_half_side + R20, bulge_90_deg_cw), # 18 (R20 fillet)
        (-central_half_side, half_height - chamfer_size, 0), # 19
        (-central_half_side - chamfer_size, half_height, 0), # 20
        (central_half_side + chamfer_size, half_height, 0), # 21
        (central_half_side, half_height - chamfer_size, 0), # 22
        (central_half_side, central_half_side + R20, bulge_90_deg_cw), # 23 (R20 fillet)
        (half_width - chamfer_size, central_half_side, 0) # Close to 1
    ]
    msp.add_lwpolyline(outer_points, close=True, dxfattribs={"layer": "GEOMETRY"})

    # Inner profile (clockwise)
    inner_points = [
        (central_half_side + R12, central_half_side, 0), # 1
        (central_half_side, central_half_side + R12, bulge_90_deg_cw), # 2 (R12 fillet)
        (central_half_side, central_half_side + R12, 0), # 3
        (central_half_side, -central_half_side - R12, 0), # 4
        (central_half_side + R12, -central_half_side, bulge_90_deg_cw), # 5 (R12 fillet)
        (central_half_side + R12, -central_half_side, 0), # 6
        (-central_half_side - R12, -central_half_side, 0), # 7
        (-central_half_side, -central_half_side - R12, bulge_90_deg_cw), # 8 (R12 fillet)
        (-central_half_side, -central_half_side - R12, 0), # 9
        (-central_half_side, central_half_side + R12, 0), # 10
        (-central_half_side - R12, central_half_side, bulge_90_deg_cw), # 11 (R12 fillet)
        (-central_half_side - R12, central_half_side, 0), # 12
        (central_half_side + R12, central_half_side, 0) # Close to 1
    ]
    msp.add_lwpolyline(inner_points, close=True, dxfattribs={"layer": "GEOMETRY"})

    # Holes
    # M24x55 (horizontal slots) - 2x
    draw_slotted_hole(msp, 0, 125, 55, 24, 0) # Top arm
    draw_slotted_hole(msp, 0, -125, 55, 24, 0) # Bottom arm

    # M20x50 (vertical slots) - 2x
    draw_slotted_hole(msp, 125, 0, 50, 20, 90) # Right arm
    draw_slotted_hole(msp, -125, 0, 50, 20, 90) # Left arm

    # M16x45 (horizontal slots) - 2x
    draw_slotted_hole(msp, 0, 250, 45, 16, 0) # Top arm, further out
    draw_slotted_hole(msp, 0, -250, 45, 16, 0) # Bottom arm, further out

    # Annotations
    msp.add_mtext("PART 701 - Steigbügel", dxfattribs={"layer": "ANNOTATION"}).set_location((0, half_height + 50), rotation=0, attachment_point=5)
    msp.add_mtext("Overall: 375 x 437 mm", dxfattribs={"layer": "ANNOTATION"}).set_location((0, half_height + 40), rotation=0, attachment_point=5)
    msp.add_mtext("Material: G75", dxfattribs={"layer": "ANNOTATION"}).set_location((0, half_height + 30), rotation=0, attachment_point=5)
    msp.add_mtext("Holes: 2x M24x55, 2x M20x50, 2x M16x45 slotted", dxfattribs={"layer": "ANNOTATION"}).set_location((0, half_height + 20), rotation=0, attachment_point=5)
    msp.add_mtext("Chamfers: 10x10, Fillets: R12, R20", dxfattribs={"layer": "ANNOTATION"}).set_location((0, half_height + 10), rotation=0, attachment_point=5)

    doc.saveas(output_filename)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <output_dxf_filename>")
        sys.exit(1)
    create_part_dxf(sys.argv[1])