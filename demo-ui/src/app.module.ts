import { Module } from "@nestjs/common";
import { ServeStaticModule } from "@nestjs/serve-static";
import { join } from "path";
import { ReviewController } from "./review/review.controller";
import { ReviewService } from "./review/review.service";

@Module({
  imports: [
    ServeStaticModule.forRoot({
      rootPath: join(__dirname, "..", "public"),
    }),
  ],
  controllers: [ReviewController],
  providers: [ReviewService],
})
export class AppModule {}
