def generate_abaqus_inp(filename):
    with open(filename, "w") as f:
        f.write("** Abaqus Input File: Two Squares with Interface Elements\n")
        
        # Part 1 (Bottom Square)
        f.write("*Part, name=PART1\n")
        f.write("*Node\n")
        node_id = 1
        for j in range(11):
            for i in range(11):
                f.write(f"{node_id}, {i/10.0}, {j/10.0}, 0.0\n")
                node_id += 1
        f.write("*Element, type=CPS4, elset=PART1_ELEM\n")
        elem_id = 1
        for j in range(10):
            for i in range(10):
                n1 = j * 11 + i + 1
                n2 = n1 + 1
                n3 = n1 + 12
                n4 = n1 + 11
                f.write(f"{elem_id}, {n1}, {n2}, {n3}, {n4}\n")
                elem_id += 1
        f.write("*End Part\n\n")
        
        # Part 2 (Top Square)
        f.write("*Part, name=PART2\n")
        f.write("*Node\n")
        node_id = 101
        for j in range(11, 22):
            for i in range(11):
                f.write(f"{node_id}, {i/10.0}, {j/10.0}, 0.0\n")
                node_id += 1
        f.write("*Element, type=CPS4, elset=PART2_ELEM\n")
        for j in range(10):
            for i in range(10):
                n1 = 101 + j * 11 + i
                n2 = n1 + 1
                n3 = n1 + 12
                n4 = n1 + 11
                f.write(f"{elem_id}, {n1}, {n2}, {n3}, {n4}\n")
                elem_id += 1
        f.write("*End Part\n\n")
        
        # Interface Elements
        f.write("*Part, name=INTERFACE\n")
        f.write("*Node\n")
        for i in range(11):
            f.write(f"{200+i}, {i/10.0}, 1.0, 0.0\n")
        f.write("*Element, type=INTERFACE_ELEM, elset=INTERFACE_SET\n")
        for i in range(10):
            f.write(f"{elem_id}, {200+i}, {200+i+1}, {100+i}, {101+i}\n")
            elem_id += 1
        f.write("*End Part\n\n")
        
        # Assembly
        f.write("*Assembly, name=ASSEMBLY\n")
        f.write("*Instance, name=PART1_INST, part=PART1\n")
        f.write("*Instance, name=PART2_INST, part=PART2\n")
        f.write("*Instance, name=INTERFACE_INST, part=INTERFACE\n")
        f.write("*End Assembly\n\n")
        
        # Boundary Conditions
        f.write("*Boundary\n")
        f.write("PART1_BOTTOM, 1, 1, -0.1\n")
        f.write("PART1_BOTTOM, 2, 2, 0\n")
        f.write("PART2_TOP, 1, 1, 0.1\n")
        f.write("PART2_TOP, 2, 2, 0\n")
        f.write("PART1_LEFT, 2, 2, 0\n")
        f.write("PART2_LEFT, 2, 2, 0\n")
        f.write("PART1_RIGHT, 2, 2, 0\n")
        f.write("PART2_RIGHT, 2, 2, 0\n")
    
# Generate the Abaqus input file
generate_abaqus_inp("interface_model.inp")

