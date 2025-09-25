//
//  ContentView.swift
//  ImageClassifier
//
//  Created by Pau Labarta Bajo on 18/9/25.
//

import SwiftUI


struct ContentView: View {
    @Environment(ClassificationModel.self) private var model
    @State private var selectedImage: String = ""
    @State private var predictionResult: String = "{}"
    
    let imageNames = ["2df3", "dog", "truck", "laptop"]
    
    var body: some View {
            VStack {
                if !model.isModelLoading {
                    
                    Text("Image classifier")
                        .font(.title)
                    
                    // Image Gallery
                    let columns = Array(repeating: GridItem(.flexible()), count: 3)
                    LazyVGrid(columns: columns, spacing: 10) {
                        ForEach(imageNames, id: \.self) { imageName in
                            // ImageTile(imageName: imageName)
                            ImageTile(imageName: imageName) {
                                selectImage(imageName)
                            }
                        }
                    }
                    .padding()
                    
                    VStack(alignment: .leading) {
                        Text("Prediction Result:")
                            .font(.headline)
                            .padding(.horizontal)
                        
                        ScrollView {
                            //Text(predictionResult)
                            //Text(selectedImage)
                            Text(model.outputText)
                                .font(.system(.body, design: .monospaced))
                                .padding()
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .frame(height: 150)
                        .background(Color.gray.opacity(0.1))
                        .cornerRadius(8)
                        .padding(.horizontal)
                    }
                } else {
                    ProgressView("Loading the model...")
                        .task {
                            await model.setupModel()
                        }
                    
                    
                    
                }
        }
        .padding()
        
    }
    
    func selectImage(_ imageName: String) {
        selectedImage = imageName
        
        Task {
            await model.predict(imageName)
        }
        
        
    }
}


struct ImageTile: View {
    let imageName: String
    let onTap: () -> Void
    
    var body: some View {
        Image(imageName)
            .resizable()
            .aspectRatio(contentMode: .fill)
            .frame(width: 100, height: 100)
            .clipped()
            .onTapGesture {
                onTap()
            }
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(.gray, lineWidth: 1)
            )
    }
}

#Preview {
    ContentView()
        .environment(ClassificationModel())
}
