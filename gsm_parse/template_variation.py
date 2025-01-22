
#### Tree Logging Calculation
question_wording = {
    "LumberYard":
        """Question: At {name}'s lumber yard, they process different types of trees. A {pine} tree yields {pine_logs} pieces of lumber, a {maple} tree yields {maple_logs} pieces, and a {walnut} tree yields {walnut_logs} pieces. Today they processed {pine_count} {pine} trees, {maple_count} {maple} trees, and {walnut_count} {walnut} trees. How many total pieces of lumber did they produce?"""
    ,
    "ForestManagement":
        """Question: {name} is a forest manager tracking timber production. When harvested, each {pine} tree produces {pine_logs} usable sections, each {maple} tree produces {maple_logs} sections, and each {walnut} tree produces {walnut_logs} sections. If their latest harvest included {pine_count} {pine} trees, {maple_count} {maple} trees, and {walnut_count} {walnut} trees, what was the total number of sections produced?"""
    ,
    "ConstructionSupply":
        """Question: {name}'s construction supply company processes trees into building materials. They can get {pine_logs} beams from each {pine} tree, {maple_logs} beams from each {maple} tree, and {walnut_logs} beams from each {walnut} tree. If they process {pine_count} {pine} trees, {maple_count} {maple} trees, and {walnut_count} {walnut} trees, how many beams will they have in total?"""
    ,
    "FurnitureMaking":
        """Question: {name} owns a furniture workshop where they cut trees into workable pieces. From each {pine} tree they get {pine_logs} pieces, from each {maple} tree they get {maple_logs} pieces, and from each {walnut} tree they get {walnut_logs} pieces. If they cut {pine_count} {pine} trees, {maple_count} {maple} trees, and {walnut_count} {walnut} trees, how many total pieces do they have?"""
    ,
    "Simplified":
        """Question: {name} needs to calculate their wood inventory. Each {pine} gives {pine_logs} units, each {maple} gives {maple_logs} units, and each {walnut} gives {walnut_logs} units. With {pine_count} {pine}, {maple_count} {maple}, and {walnut_count} {walnut}, how many total units are there?"""
    
}


plural_wording = {
    "LumberYard":"pieces",
    "ForestManagement": "sections",
    "ConstructionSupply": "beams",
    "FurnitureMaking": "pieces",
    "Simplified": "units",
}

