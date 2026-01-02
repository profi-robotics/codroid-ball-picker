// Robot calibration plate for Codroid (units in mm).
// export_part: "combined" (default), "base", or "inlay".
// Marker centers follow SVG-style coordinates: origin at top-left, +x right, +y down.
export_part = "combined";
plate_width_mm = 297.0;
plate_height_mm = 210.0;
plate_thickness_mm = 5;
marker_size_mm = 50.0;
marker_recess_mm = 0;
marker_grid_cells = 6;
marker_inlay_mm = 0.6;
marker_cell_size_mm = marker_size_mm / marker_grid_cells;
label_size_mm = 8;
label_depth_mm = 0.6;
label_offset_mm = 8.0;
margin_bottom_mm = 20.0;
center_mark_length_mm = 6;
center_mark_gap_mm = 1;
center_mark_width_mm = 0.6;
center_mark_depth_mm = 0.6;
footer_text = "Estun Codroid Coordinate System calibration board by Profi Robotics Ukraine";
footer_text_size_mm = 4;
footer_text_depth_mm = 0.6;
base_color = [1, 1, 1];
inlay_color = [0, 0, 0];

marker_centers_mm = [
  [10, 35.0, 35.0],
  [11, 262.0, 35.0],
  [12, 262.0, 165.0],
];

robot_sequence = [
  [10, 1],
  [11, 2],
  [12, 3],
];

function to_scad(point_xy) = [point_xy[0], plate_height_mm - point_xy[1]];
function center_for(marker_id) = [for (marker = marker_centers_mm) if (marker[0] == marker_id) [marker[1], marker[2]]][0];

black_cells_id10 = [
  [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
  [1, 0], [1, 5],
  [2, 0], [2, 2], [2, 3], [2, 5],
  [3, 0], [3, 2], [3, 3], [3, 5],
  [4, 0], [4, 1], [4, 2], [4, 3], [4, 5],
  [5, 0], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5]
];

black_cells_id11 = [
  [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
  [1, 0], [1, 1], [1, 2], [1, 3], [1, 5],
  [2, 0], [2, 1], [2, 2], [2, 3], [2, 5],
  [3, 0], [3, 2], [3, 4], [3, 5],
  [4, 0], [4, 1], [4, 5],
  [5, 0], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5]
];

black_cells_id12 = [
  [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
  [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5],
  [2, 0], [2, 4], [2, 5],
  [3, 0], [3, 2], [3, 5],
  [4, 0], [4, 1], [4, 5],
  [5, 0], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5]
];

function black_cells_for(marker_id) =
  marker_id == 10 ? black_cells_id10 :
  marker_id == 11 ? black_cells_id11 :
  marker_id == 12 ? black_cells_id12 : [];

module marker_recess(center_xy) {
  let(scad_xy = to_scad(center_xy))
    translate([
      scad_xy[0] - marker_size_mm / 2,
      scad_xy[1] - marker_size_mm / 2,
      plate_thickness_mm - marker_recess_mm
    ])
      cube([marker_size_mm, marker_size_mm, marker_recess_mm + 0.1]);
}

module marker_black_tiles(center_xy, marker_id, tile_height_mm, z_offset_mm) {
  marker_origin_x = center_xy[0] - marker_size_mm / 2;
  marker_origin_y = center_xy[1] - marker_size_mm / 2;
  for (cell = black_cells_for(marker_id)) {
    row = cell[0];
    col = cell[1];
    tile_x = marker_origin_x + col * marker_cell_size_mm;
    tile_y = marker_origin_y + row * marker_cell_size_mm;
    translate([tile_x, plate_height_mm - (tile_y + marker_cell_size_mm), z_offset_mm])
      cube([marker_cell_size_mm, marker_cell_size_mm, tile_height_mm]);
  }
}

function label_y_for(center_xy) =
  (center_xy[1] + marker_size_mm / 2 + label_offset_mm > plate_height_mm - margin_bottom_mm) ?
  (center_xy[1] - marker_size_mm / 2 - label_offset_mm) :
  (center_xy[1] + marker_size_mm / 2 + label_offset_mm);

module sequence_label(center_xy, label_text) {
  label_xy = [center_xy[0], label_y_for(center_xy)];
  scad_xy = to_scad(label_xy);
  translate([scad_xy[0], scad_xy[1], plate_thickness_mm - label_depth_mm])
    linear_extrude(height=label_depth_mm + 0.1)
      text(
        text = str(label_text),
        size = label_size_mm,
        font = "Liberation Sans:style=Bold",
        halign = "center",
        valign = "center"
      );
}

module center_marks(center_xy) {
  left_x0 = center_xy[0] - marker_size_mm / 2 - center_mark_length_mm;
  left_x1 = center_xy[0] - marker_size_mm / 2 - center_mark_gap_mm;
  right_x0 = center_xy[0] + marker_size_mm / 2 + center_mark_gap_mm;
  right_x1 = center_xy[0] + marker_size_mm / 2 + center_mark_length_mm;
  top_y0 = center_xy[1] - marker_size_mm / 2 - center_mark_length_mm;
  top_y1 = center_xy[1] - marker_size_mm / 2 - center_mark_gap_mm;
  bottom_y0 = center_xy[1] + marker_size_mm / 2 + center_mark_gap_mm;
  bottom_y1 = center_xy[1] + marker_size_mm / 2 + center_mark_length_mm;
  horizontal_segments = [
    [[left_x0, center_xy[1]], [left_x1, center_xy[1]]],
    [[right_x0, center_xy[1]], [right_x1, center_xy[1]]] 
  ];
  vertical_segments = [
    [[center_xy[0], top_y0], [center_xy[0], top_y1]],
    [[center_xy[0], bottom_y0], [center_xy[0], bottom_y1]]
  ];
  for (segment = horizontal_segments) {
    start_xy = segment[0];
    end_xy = segment[1];
    scad_xy = to_scad([start_xy[0], start_xy[1]]);
    translate([scad_xy[0], scad_xy[1] - center_mark_width_mm / 2, plate_thickness_mm - center_mark_depth_mm])
      cube([end_xy[0] - start_xy[0], center_mark_width_mm, center_mark_depth_mm + 0.1]);
  }
  for (segment = vertical_segments) {
    start_xy = segment[0];
    end_xy = segment[1];
    y_min = min(start_xy[1], end_xy[1]);
    y_max = max(start_xy[1], end_xy[1]);
    scad_xy = to_scad([start_xy[0], y_max]);
    translate([scad_xy[0] - center_mark_width_mm / 2, scad_xy[1], plate_thickness_mm - center_mark_depth_mm])
      cube([center_mark_width_mm, y_max - y_min, center_mark_depth_mm + 0.1]);
  }
}

module footer_label() {
  footer_xy = [plate_width_mm / 2, plate_height_mm - margin_bottom_mm / 2];
  scad_xy = to_scad(footer_xy);
  translate([scad_xy[0], scad_xy[1], plate_thickness_mm - footer_text_depth_mm])
    linear_extrude(height=footer_text_depth_mm + 0.1)
      text(
        text = footer_text,
        size = footer_text_size_mm,
        font = "Liberation Sans:style=Bold",
        halign = "center",
        valign = "center"
      );
}

module base_plate() {
  difference() {
    cube([plate_width_mm, plate_height_mm, plate_thickness_mm]);
    for (marker = marker_centers_mm) {
      if (marker_recess_mm > 0) {
        marker_recess([marker[1], marker[2]]);
      }
      center_marks([marker[1], marker[2]]);
      if (marker_inlay_mm > 0) {
        marker_black_tiles([marker[1], marker[2]], marker[0], marker_inlay_mm + 0.1, plate_thickness_mm - marker_inlay_mm);
      }
    }
    for (entry = robot_sequence) {
      sequence_label(center_for(entry[0]), entry[1]);
    }
    footer_label();
  }
}

module inlay_tiles() {
  if (marker_inlay_mm > 0) {
    for (marker = marker_centers_mm) {
      marker_black_tiles([marker[1], marker[2]], marker[0], marker_inlay_mm, plate_thickness_mm - marker_inlay_mm);
    }
  }
}

if (export_part == "base") {
  color(base_color) base_plate();
} else if (export_part == "inlay") {
  color(inlay_color) inlay_tiles();
} else {
  color(base_color) base_plate();
  color(inlay_color) inlay_tiles();
}
