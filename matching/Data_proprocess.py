

def main():
    pass



if __name__ == "__main__":
    seqs = ['c041', 'c042', 'c043', 'c044', 'c045', 'c046']
    # seqs = ['c041'] 
    for seq in seqs:
        print(f'Start processing {seq} ---')
        main(seq)
        print(f'Camera {seq} finished')
