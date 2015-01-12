package main

import (
	"bufio"
	"encoding/binary"
	"errors"
	"github.com/boltdb/bolt"
	"log"
	"os"
)

func uint32ToBytes(n uint32) []byte {
	p := make([]byte, 4)
	binary.LittleEndian.PutUint32(p, n)
	return p
}

func uint32FromBytes(p []byte) uint32 {
	return binary.LittleEndian.Uint32(p)
}

func makeByteBucket(db *bolt.DB, bucketName string, num, channel, height, width uint32) {
	db.Update(func(tx *bolt.Tx) error {
		_, err := tx.CreateBucket([]byte(bucketName))
		if err != nil {
			return err
		}
		bucketDim, err := tx.CreateBucket([]byte(bucketName + "_dim"))
		if err != nil {
			return err
		}
		bucketDim.Put([]byte("type"), []byte("byte"))
		bucketDim.Put([]byte("num"), uint32ToBytes(num))
		bucketDim.Put([]byte("channel"), uint32ToBytes(channel))
		bucketDim.Put([]byte("height"), uint32ToBytes(height))
		bucketDim.Put([]byte("width"), uint32ToBytes(width))
		return nil
	})
}

func read(in *bufio.Reader, p []byte) error {
	var err error
	for i := 0; i < len(p); i++ {
		p[i], err = in.ReadByte()
		if err != nil {
			return err
		}
	}
	return nil
}

func readInt(in *bufio.Reader) (uint32, error) {
	p := make([]byte, 4)
	err := read(in, p)
	if err != nil {
		return 0, err
	}
	return binary.BigEndian.Uint32(p), nil
}

func loadImages(db *bolt.DB, file string) error {
	f, err := os.Open(file)
	if err != nil {
		return err
	}

	// check magic number
	in := bufio.NewReader(f)

	magicNumber, err := readInt(in)
	if err != nil {
		return err
	}
	if magicNumber != 2051 {
		return errors.New("invalid magic number for image file")
	}

	numItems, err := readInt(in)
	if err != nil {
		return err
	}
	log.Printf("Image file contains %d images\n", numItems)

	rows, err := readInt(in)
	if err != nil {
		return err
	}
	cols, err := readInt(in)
	if err != nil {
		return err
	}
	log.Printf("Image file rows=%d, cols=%d\n", rows, cols)

	makeByteBucket(db, "images", numItems, uint32(1), uint32(rows), uint32(cols))

	pixels := make([]byte, rows*cols)
	for index := uint32(0); index < numItems; index++ {
		db.Update(func(tx *bolt.Tx) error {
			bucket := tx.Bucket([]byte("images"))
			err := read(in, pixels)
			if err != nil {
				return err
			}
			bucket.Put(uint32ToBytes(index), pixels)
			return nil
		})
	}
	log.Printf("Loaded %d images into the database\n", numItems)

	return nil
}

func loadLabels(db *bolt.DB, file string) error {
	f, err := os.Open(file)
	if err != nil {
		return err
	}

	// check magic number
	in := bufio.NewReader(f)

	magicNumber, err := readInt(in)
	if err != nil {
		return err
	}
	if magicNumber != 2049 {
		return errors.New("invalid magic number for label file")
	}

	numItems, err := readInt(in)
	if err != nil {
		return err
	}
	log.Printf("Label file contains %d labels\n", numItems)

	makeByteBucket(db, "labels", numItems, uint32(1), uint32(1), uint32(1))

	label := make([]byte, 1)
	for index := uint32(0); index < numItems; index++ {
		db.Update(func(tx *bolt.Tx) error {
			bucket := tx.Bucket([]byte("labels"))
			err := read(in, label)
			if err != nil {
				return err
			}
			bucket.Put(uint32ToBytes(index), label)
			return nil
		})
	}
	log.Printf("Loaded %d labels into the database\n", numItems)

	return nil
}

func main() {
	args := os.Args[1:]
	if len(args) != 3 {
		log.Fatalln("Three program arguments needed: <image file> <label file> <db file>")
	}

	imageFile := args[0]
	labelFile := args[1]
	dbFile := args[2]

	log.Printf("Creating bolt db file '%s' with image file '%s' and label file '%s'\n", dbFile, imageFile, labelFile)

	db, err := bolt.Open(dbFile, 0600, nil)
	if err != nil {
		log.Fatalf("Failed to open db file: error='%s'\n", err)
	}
	defer db.Close()

	err = loadImages(db, imageFile)
	if err != nil {
		log.Fatalf("Failed to load images into database: error='%s'\n", err)
	}

	err = loadLabels(db, labelFile)
	if err != nil {
		log.Fatalf("Failed to load labels into database: error='%s'\n", err)
	}
}
